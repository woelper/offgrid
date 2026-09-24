/* A small C surface over stable-diffusion.h.
 *
 * sd_ctx_params_t and sd_img_gen_params_t carry tens of fields each, nested
 * several deep, and are filled by the library's own *_init functions. Mirroring
 * those layouts in Rust would mean re-deriving them by hand every time sd.cpp
 * adds a field, with the failure mode being writes at the wrong offset rather
 * than a compile error. This shim keeps the structs on the C side, where the
 * headers define them, and hands Rust six plain functions. */
#ifndef OFFGRID_SD_H
#define OFFGRID_SD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* step is 1-based; steps is the total for this sampling pass. */
typedef void (*offgrid_sd_progress_cb)(int step, int steps, void *data);
/* RGB rows, tightly packed. Borrowed for the duration of the call. */
typedef void (*offgrid_sd_preview_cb)(int width, int height,
                                      const unsigned char *rgb, void *data);

/* Load a model. Either model_path names a single checkpoint (SD 1.5), or the
 * three split paths do (Z-Image and the other recent models ship the diffusion
 * model, the VAE and the text encoder separately). Pass NULL for what does not
 * apply. Returns NULL on failure. */
/* offload_to_cpu keeps the weights in RAM and streams them to the accelerator
 * as each is needed: the only way a 10 GB model runs on an 8 GB card. It is
 * what the sd-cli flag of that name does, expressed as a backend assignment.
 * On a CPU-only build it changes nothing, the weights already being in RAM. */
void *offgrid_sd_new(const char *model_path, const char *diffusion_path,
                     const char *vae_path, const char *llm_path, int n_threads,
                     int flash_attn, int offload_to_cpu);
void offgrid_sd_free(void *ctx);

/* Generate one image. On success returns 1 and fills *out_rgb with a buffer
 * owned by the caller (release it with offgrid_sd_free_buf), plus its size. */
/* sampler is sd.cpp's sample_method_t; -1 keeps the library's default. Models
 * are distilled for particular samplers — Qwen-Image wants euler — and the
 * wrong one produces noise rather than an error. */
int offgrid_sd_generate(void *ctx, const char *prompt, const char *negative,
                        int steps, int width, int height, float cfg,
                        int64_t seed, int sampler, unsigned char **out_rgb,
                        int *out_width, int *out_height);
void offgrid_sd_free_buf(unsigned char *buf);

/* Warnings and errors from sd.cpp, so a failure can say what went wrong
 * instead of pointing at a log nobody is reading. level is sd_log_level_t. */
typedef void (*offgrid_sd_log_cb)(int level, const char *text, void *data);
void offgrid_sd_set_log(offgrid_sd_log_cb cb, void *data);

/* Callbacks are global in sd.cpp, and so are these. `data` is passed back
 * untouched. Pass NULL to clear. */
void offgrid_sd_set_progress(offgrid_sd_progress_cb cb, void *data);
/* interval is in denoiser steps; previews are decoded with the cheap
 * projection, not the VAE. */
void offgrid_sd_set_preview(offgrid_sd_preview_cb cb, int interval, void *data);

#ifdef __cplusplus
}
#endif

#endif /* OFFGRID_SD_H */
