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
    /* A VAE-decoded preview of a model that makes alpha has four channels,
     * like its finished image; the projection kind has three. */
    if (frame->data == NULL || (frame->channel != 3 && frame->channel != 4)) {
        return;
    }
    g_preview_cb((int)frame->width, (int)frame->height, (int)frame->channel,
                 frame->data, g_preview_data);
}

void offgrid_sd_set_progress(offgrid_sd_progress_cb cb, void *data) {
    g_progress_cb = cb;
    g_progress_data = data;
    sd_set_progress_callback(cb ? progress_trampoline : NULL, NULL);
}

void offgrid_sd_set_preview(offgrid_sd_preview_cb cb, int mode, int interval,
                            void *data) {
    g_preview_cb = cb;
    g_preview_data = data;
    if (mode < 0 || mode >= PREVIEW_COUNT) {
        mode = PREVIEW_NONE;
    }
    /* denoised=true, noisy=false keeps the picture from flickering with the
     * sampler's noise. */
    sd_set_preview_callback(cb ? preview_trampoline : NULL,
                            (enum preview_t)mode, interval, true, false, NULL);
}

/* Treat an empty string as absent: it is easier to pass "" from Rust than to
 * juggle null pointers, and sd.cpp wants NULL for a path it should ignore. */
static const char *or_null(const char *s) {
    return (s && s[0]) ? s : NULL;
}

void *offgrid_sd_new(const char *model_path, const char *diffusion_path,
                     const char *vae_path, const char *llm_path,
                     const char *llm_vision_path, int n_threads, int flash_attn,
                     int offload_to_cpu) {
    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    params.model_path = or_null(model_path);
    params.diffusion_model_path = or_null(diffusion_path);
    params.vae_path = or_null(vae_path);
    params.llm_path = or_null(llm_path);
    params.llm_vision_path = or_null(llm_vision_path);
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

void offgrid_sd_cancel(void *ctx, int reset) {
    if (ctx) {
        sd_cancel_generation((sd_ctx_t *)ctx,
                             reset ? SD_CANCEL_RESET : SD_CANCEL_ALL);
    }
}

void offgrid_sd_free(void *ctx) {
    if (ctx) {
        free_sd_ctx((sd_ctx_t *)ctx);
    }
}

int offgrid_sd_generate(void *ctx, const char *prompt, const char *negative,
                        int steps, int width, int height, float cfg,
                        int64_t seed, int sampler,
                        const offgrid_sd_image *refs, int ref_count,
                        unsigned char **out_pixels, int *out_width,
                        int *out_height, int *out_channels) {
    if (!ctx || !out_pixels) {
        return 0;
    }
    *out_pixels = NULL;

    sd_img_gen_params_t params;
    sd_img_gen_params_init(&params);
    params.prompt = prompt;
    params.negative_prompt = negative ? negative : "";
    params.width = width;
    params.height = height;
    params.seed = seed;
    params.batch_count = 1;
    /* sd_image_t is laid out differently from our own descriptor, so the
     * references are repacked here. The library reads the pixels but does not
     * take them: the buffers are the caller's and outlive this call. The cap is
     * this array, not the library — a request for more than it holds would
     * otherwise scribble past the end. */
    sd_image_t references[OFFGRID_SD_MAX_REFS];
    if (refs && ref_count > 0) {
        if (ref_count > OFFGRID_SD_MAX_REFS) {
            ref_count = OFFGRID_SD_MAX_REFS;
        }
        int kept = 0;
        for (int i = 0; i < ref_count; i++) {
            if (!refs[i].pixels || refs[i].width <= 0 || refs[i].height <= 0) {
                continue;
            }
            references[kept].width = (uint32_t)refs[i].width;
            references[kept].height = (uint32_t)refs[i].height;
            references[kept].channel = (uint32_t)refs[i].channels;
            references[kept].data = (uint8_t *)refs[i].pixels;
            kept++;
        }
        if (kept > 0) {
            params.ref_images = references;
            params.ref_images_count = kept;
        }
    }
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

    /* Copy the pixels out as they came and hand that back, so ownership on the
     * Rust side is one plain malloc'd block with no sd_image_t attached. Models
     * differ on channels — Qwen-Image 2.1 returns RGBA where SD 1.5 and
     * Z-Image return RGB — and the count travels with the buffer rather than
     * the alpha being thrown away here. */
    sd_image_t *image = &images[0];
    size_t pixel_count = (size_t)image->width * image->height;
    size_t bytes = pixel_count * image->channel;
    int ok = 0;
    if (!image->data || bytes == 0) {
        complain("the generated image is empty");
    } else if (image->channel != 3 && image->channel != 4) {
        static char note[96];
        snprintf(note, sizeof(note), "expected RGB or RGBA, got %u channels",
                 image->channel);
        complain(note);
    } else {
        unsigned char *copy = (unsigned char *)malloc(bytes);
        if (!copy) {
            complain("out of memory copying the image");
        } else {
            memcpy(copy, image->data, bytes);
            *out_pixels = copy;
            if (out_width) {
                *out_width = (int)image->width;
            }
            if (out_height) {
                *out_height = (int)image->height;
            }
            if (out_channels) {
                *out_channels = (int)image->channel;
            }
            ok = 1;
        }
    }
    free_sd_images(images, count);
    return ok;
}

void offgrid_sd_free_buf(unsigned char *buf) { free(buf); }
