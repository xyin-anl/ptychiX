from abc import ABC, abstractmethod
from dataclasses import dataclass
import cmath

import torch
from torch.fft import fft2, fftfreq, fftshift, ifft2, ifftshift

from .typing import ComplexTensor, RealTensor


@dataclass(frozen=True)
class WavefieldPropagatorParameters:
    width_px: int
    """number of pixels in the x-direction"""
    height_px: int
    """number of pixels in the y-direction"""
    pixel_width_wlu: float
    """pixel width in wavelengths"""
    pixel_aspect_ratio: float
    """pixel aspect ratio (width / height)"""
    propagation_distance_wlu: float
    """propagation distance in wavelengths"""

    @classmethod
    def create_simple(
        cls,
        wavelength_m: float,
        width_px: int,
        height_px: int,
        pixel_width_m: float,
        pixel_height_m: float,
        propagation_distance_m: float,
    ) -> WavefieldPropagatorParameters:
        """Creates propagator paramaters dataclass from quantities with physical length units.
        :param wavelength_m: illumination wavelength in meters
        :type wavelength_m: float
        :param width_px: number of pixels in the x-direction
        :type width_px: int
        :param height_px: number of pixels in the y-direction
        :type height_px: int
        :param pixel_width_m: source plane pixel width in meters
        :type pixel_width_m: float
        :param pixel_height_m: source plane pixel height in meters
        :type pixel_height_m: float
        :param propagation_distance_m: propagation distance in meters
        :type propagation_distance_m: float
        :returns: a dataclass that contains the nondimensionalized propagator parameters
        :rtype: WavefieldPropagatorParameters
        """
        return cls(
            width_px,
            height_px,
            pixel_width_m / wavelength_m,
            pixel_width_m / pixel_height_m,
            propagation_distance_m / wavelength_m,
        )

    @property
    def fresnel_number(self) -> float:
        pixel_width_wlu_sq = self.pixel_width_wlu * self.pixel_width_wlu
        return pixel_width_wlu_sq / self.propagation_distance_wlu

    def get_spatial_coordinates(self) -> tuple[RealTensor, RealTensor]:
        ii = torch.arange(self.width_px)
        jj = torch.arange(self.height_px)
        JJ, II = torch.meshgrid(jj, ii)  # FIXME ij or xy?
        XX = II - self.width_px // 2
        YY = JJ - self.height_px // 2
        return YY, XX

    def get_frequency_coordinates(self) -> tuple[RealTensor, RealTensor]:
        fx = fftshift(fftfreq(self.width_px))
        fy = fftshift(fftfreq(self.height_px))
        FY, FX = torch.meshgrid(fy, fx)  # FIXME ij or xy?
        return FY, FX


class WavefieldPropagator(ABC):

    @abstractmethod
    def propagate_forward(self, wavefield: ComplexTensor) -> ComplexTensor:
        pass

    @abstractmethod
    def propagate_backward(self, wavefield: ComplexTensor) -> ComplexTensor:
        pass


class FourierPropagator(WavefieldPropagator):

    def propagate_forward(self, wavefield: ComplexTensor) -> ComplexTensor:
        return fftshift(fft2(ifftshift(wavefield)))

    def propagate_backward(self, wavefield: ComplexTensor) -> ComplexTensor:
        return fftshift(ifft2(ifftshift(wavefield)))


class AngularSpectrumPropagator(WavefieldPropagator):

    def __init__(self, parameters: WavefieldPropagatorParameters) -> None:
        ar = parameters.pixel_aspect_ratio

        i2piz = 2j * torch.pi * parameters.propagation_distance_wlu
        FY, FX = parameters.get_frequency_coordinates()
        F2 = torch.square(FX) + torch.square(ar * FY)
        ratio = F2 / (parameters.pixel_width_wlu**2)
        tf = torch.exp(i2piz * torch.sqrt(1 - ratio))

        self._transfer_function = torch.where(ratio < 1, tf, 0)

    def propagate(self, wavefield: ComplexTensor) -> ComplexTensor:
        return fftshift(ifft2(self._transfer_function * fft2(ifftshift(wavefield))))


class FresnelTransferFunctionPropagator(WavefieldPropagator):

    def __init__(self, parameters: WavefieldPropagatorParameters) -> None:
        ar = parameters.pixel_aspect_ratio

        i2piz = 2j * torch.pi * parameters.propagation_distance_wlu
        FY, FX = parameters.get_frequency_coordinates()
        F2 = torch.square(FX) + torch.square(ar * FY)
        ratio = F2 / (parameters.pixel_width_wlu**2)

        self._transfer_function = torch.exp(i2piz * (1 - ratio / 2))

    def propagate(self, wavefield: ComplexTensor) -> ComplexTensor:
        return fftshift(ifft2(self._transfer_function * fft2(ifftshift(wavefield))))


class FresnelTransformPropagator(WavefieldPropagator):

    def __init__(self, parameters: WavefieldPropagatorParameters) -> None:
        ipi = 1j * torch.pi

        Fr = parameters.fresnel_number
        ar = parameters.pixel_aspect_ratio
        N = parameters.width_px
        M = parameters.height_px
        YY, XX = parameters.get_spatial_coordinates()

        C0 = Fr / (1j * ar)
        C1 = cmath.exp(2j * cmath.pi * parameters.propagation_distance_wlu)
        C2 = torch.exp((torch.square(XX / N) + torch.square(ar * YY / M)) * ipi / Fr)
        is_forward = (parameters.propagation_distance_wlu >= 0.)

        self._is_forward = is_forward
        self._A = C2 * C1 * C0 if is_forward else C2 * C1 / C0
        self._B = torch.exp(ipi * Fr * (torch.square(XX) + torch.square(YY / ar)))

    def propagate(self, wavefield: ComplexTensor) -> ComplexTensor:
        if self._is_forward:
            return self._A * fftshift(fft2(ifftshift(wavefield * self._B)))
        else:
            return self._B * fftshift(ifft2(ifftshift(wavefield * self._A)))


class FraunhoferPropagator(WavefieldPropagator):

    def __init__(self, parameters: WavefieldPropagatorParameters) -> None:
        ipi = 1j * torch.pi

        Fr = parameters.fresnel_number
        ar = parameters.pixel_aspect_ratio
        N = parameters.width_px
        M = parameters.height_px
        YY, XX = parameters.get_spatial_coordinates()

        C0 = Fr / (1j * ar)
        C1 = cmath.exp(2j * cmath.pi * parameters.propagation_distance_wlu)
        C2 = torch.exp((torch.square(XX / N) + torch.square(ar * YY / M)) * ipi / Fr)
        is_forward = (parameters.propagation_distance_wlu >= 0.)

        self._is_forward = is_forward
        self._A = C2 * C1 * C0 if is_forward else C2 * C1 / C0

    def propagate(self, wavefield: ComplexTensor) -> ComplexTensor:
        if self._is_forward:
            return self._A * fftshift(fft2(ifftshift(wavefield)))
        else:
            return fftshift(ifft2(ifftshift(wavefield * self._A)))
