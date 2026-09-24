#include "offgrid_sd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "stable-diffusion.h"

/* sd.cpp's callbacks are process-global, so ours are too. Only one generation
 * runs at a time — the worker thread owns the context — so a pair of statics is
 * enough, and it keeps the Rust side from having to thread state through the
 * library. */
static offgrid_sd_progress_cb g_progress_cb = NULL;
static void *g_progress_data = NULL;
static offgrid_sd_preview_cb g_preview_cb = NULL;
static void *g_preview_data = NULL;

static offgrid_sd_log_cb g_log_cb = NULL;
static void *g_log_data = NULL;

static void log_trampoline(enum sd_log_level_t level, const char *text,
                           void *data) {
    (void)data;
    if (g_log_cb && text) {
        g_log_cb((int)level, text, g_log_data);
    }
}

/* Route the shim's own failures through sd.cpp's log, so the Rust side learns
 * about them the same way it learns about the library's. */
static void complain(const char *text) {
    if (g_log_cb) {
        g_log_cb((int)SD_LOG_ERROR, text, g_log_data);
    }
}

void offgrid_sd_set_log(offgrid_sd_log_cb cb, void *data) {
    g_log_cb = cb;
    g_log_data = data;
    sd_set_log_callback(cb ? log_trampoline : NULL, NULL);
}

static void progress_trampoline(int step, int steps, float time, void *data) {
    (void)time;
    (void)data;
    if (g_progress_cb) {
        g_progress_cb(step, steps, g_progress_data);
    }
}

static void preview_trampoline(int step, int frame_count, sd_image_t *frames,
                               bool is_noisy, void *data) {
    (void)step;
    (void)is_noisy;
    (void)data;
    if (!g_preview_cb || frame_count < 1 || frames == NULL) {
        return;
    }
    sd_image_t *frame = &frames[0];
    /* Previews arrive as RGB; anything else we let pass rather than guess. */
    if (frame->channel != 3 || frame->data == NULL) {
        return;
    }
    g_preview_cb((int)frame->width, (int)frame->height, frame->data,
                 g_preview_data);
}

void offgrid_sd_set_progress(offgrid_sd_progress_cb cb, void *data) {
    g_progress_cb = cb;
    g_progress_data = data;
    sd_set_progress_callback(cb ? progress_trampoline : NULL, NULL);
}

void offgrid_sd_set_preview(offgrid_sd_preview_cb cb, int interval,
                            void *data) {
    g_preview_cb = cb;
    g_preview_data = data;
    /* PREVIEW_PROJ is the linear approximation of the latents: no VAE, so it
     * costs almost nothing next to a denoiser step. denoised=true, noisy=false
     * keeps the picture from flickering with the sampler's noise. */
    sd_set_preview_callback(cb ? preview_trampoline : NULL, PREVIEW_PROJ,
                            interval, true, false, NULL);
}

/* Treat an empty string as absent: it is easier to pass "" from Rust than to
 * juggle null pointers, and sd.cpp wants NULL for a path it should ignore. */
static const char *or_null(const char *s) {
    return (s && s[0]) ? s : NULL;
}

void *offgrid_sd_new(const char *model_path, const char *diffusion_path,
                     const char *vae_path, const char *llm_path, int n_threads,
                     int flash_attn, int offload_to_cpu) {
    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    params.model_path = or_null(model_path);
    params.diffusion_model_path = or_null(diffusion_path);
    params.vae_path = or_null(vae_path);
    params.llm_path = or_null(llm_path);
    params.n_threads = n_threads;
    /* Flash attention in the diffusion model: recommended for Z-Image, and
     * what the sd.cpp docs use in their own examples. */
    params.diffusion_flash_attn = flash_attn ? true : false;
    /* What sd-cli's --offload-to-cpu expands to: every weight assigned to the
     * cpu params backend, loaded into the accelerator on demand. */
    if (offload_to_cpu) {
        params.params_backend = "*=cpu";
    }
    return (void *)new_sd_ctx(&params);
}

void offgrid_sd_free(void *ctx) {
    if (ctx) {
        free_sd_ctx((sd_ctx_t *)ctx);
    }
}

int offgrid_sd_generate(void *ctx, const char *prompt, const char *negative,
                        int steps, int width, int height, float cfg,
                        int64_t seed, int sampler, unsigned char **out_rgb,
                        int *out_width, int *out_height) {
    if (!ctx || !out_rgb) {
        return 0;
    }
    *out_rgb = NULL;

    sd_img_gen_params_t params;
    sd_img_gen_params_init(&params);
    params.prompt = prompt;
    params.negative_prompt = negative ? negative : "";
    params.width = width;
    params.height = height;
    params.seed = seed;
    params.batch_count = 1;
    params.sample_params.sample_steps = steps;
    params.sample_params.guidance.txt_cfg = cfg;
    if (sampler >= 0 && sampler < SAMPLE_METHOD_COUNT) {
        params.sample_params.sample_method = (enum sample_method_t)sampler;
    }

    sd_image_t *images = NULL;
    int count = 0;
    if (!generate_image((sd_ctx_t *)ctx, &params, &images, &count)) {
        complain("generate_image failed");
        if (images) {
            free_sd_images(images, count);
        }
        return 0;
    }
    if (images == NULL || count < 1) {
        complain("generate_image returned no images");
        return 0;
    }

    /* Copy out as packed RGB and hand that back, so ownership on the Rust side
     * is one plain malloc'd block with no sd_image_t attached. Models differ on
     * channels — Qwen-Image 2.1 returns RGBA where SD 1.5 and Z-Image return
     * RGB — and everything downstream, the texture upload and the PNG writer,
     * wants three, so any alpha is dropped here rather than everywhere. */
    sd_image_t *image = &images[0];
    size_t pixels = (size_t)image->width * image->height;
    int ok = 0;
    if (!image->data || pixels == 0) {
        complain("the generated image is empty");
    } else if (image->channel != 3 && image->channel != 4) {
        static char note[96];
        snprintf(note, sizeof(note), "expected RGB or RGBA, got %u channels",
                 image->channel);
        complain(note);
    } else {
        unsigned char *copy = (unsigned char *)malloc(pixels * 3);
        if (!copy) {
            complain("out of memory copying the image");
        } else {
            for (size_t i = 0; i < pixels; i++) {
                memcpy(copy + i * 3, image->data + i * image->channel, 3);
            }
            *out_rgb = copy;
            if (out_width) {
                *out_width = (int)image->width;
            }
            if (out_height) {
                *out_height = (int)image->height;
            }
            ok = 1;
        }
    }
    free_sd_images(images, count);
    return ok;
}

void offgrid_sd_free_buf(unsigned char *buf) { free(buf); }
