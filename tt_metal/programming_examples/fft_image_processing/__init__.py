"""fft_image_processing — modular FFT-based image processing demo.

Modules:
    image_loader         — load grayscale images, optional Gaussian noise
    fft_module           — custom 2-D FFT backed by Tenstorrent Wormhole (tt_fft)
    torch_fft_module     — PyTorch reference (torch.fft.fft2 / ifft2)
    filters              — low-pass / high-pass / band-pass masks
    metrics              — MSE, PSNR
    visualization        — matplotlib helpers for image + spectrum plots
    benchmarking         — timing across engines
    main                 — CLI entry point
"""
