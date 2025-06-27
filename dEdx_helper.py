import numpy as np
import ROOT

class dEdx_helper:
    # Define physical constants in general and for liquid argon
    charge = 1
    rho = 1.39  # g/cm^3
    K = 0.307075  # MeV cm^2 / mol
    Z = 18
    A = 39.948  # g / mol
    I = 188.0e-6  # MeV, mean excitation energy
    me = 0.511  # MeV

    def __init__(self, mass, KE, pitch):
        self.mass = mass
        self.KE = KE
        self.pitch = pitch

    def rel_gamma(self):
        this_gamma = (self.KE / self.mass) + 1.0
        return this_gamma

    def rel_beta2(self):
        gamma = self.rel_gamma()
        this_beta2 = 1 - (1.0 / (gamma * gamma))
        return this_beta2

    def Landau_xi(self):
        beta2 = self.rel_beta2()
        xi = dEdx_helper.rho * self.pitch * 0.5 * dEdx_helper.K * (dEdx_helper.Z / dEdx_helper.A) / beta2
        return xi

    def Get_Wmax(self):
        gamma = self.rel_gamma()
        beta2 = self.rel_beta2()
        Wmax = (2.0 * dEdx_helper.me * beta2 * gamma * gamma) / (1.0 + 2.0 * dEdx_helper.me * (gamma / self.mass) + (dEdx_helper.me / self.mass) * (dEdx_helper.me / self.mass))
        return Wmax

    def densityEffect(self):
        gamma = self.rel_gamma()
        beta = np.sqrt(self.rel_beta2())
        density_y = np.log10(beta * gamma)
        ln10 = np.log(10)
        if density_y > 3.0:
            this_delta = 2.0 * ln10 * density_y - 5.2146
        elif density_y < 0.2:
            this_delta = 0
        else:
            this_delta = 2.0 * ln10 * density_y - 5.2146 + 0.19559 * np.power(3.0 - density_y, 3)
        return this_delta

    def dEdx_mean(self):  # Bethe-Bloch
        gamma = self.rel_gamma()
        beta = np.sqrt(self.rel_beta2())
        Wmax = self.Get_Wmax()
        dEdx = (dEdx_helper.rho * dEdx_helper.K * dEdx_helper.Z * dEdx_helper.charge * dEdx_helper.charge) / (dEdx_helper.A * beta * beta) * \
               (0.5 * np.log(2 * dEdx_helper.me * gamma * gamma * beta * beta * Wmax / (dEdx_helper.I * dEdx_helper.I)) - beta * beta - self.densityEffect() / 2)
        return dEdx

    def dEdx_MPV(self):
        gamma = self.rel_gamma()
        beta = np.sqrt(self.rel_beta2())
        delta = self.densityEffect()
        xi = self.Landau_xi()
        a0 = 2 * dEdx_helper.me * (beta * gamma) * (beta * gamma) / dEdx_helper.I
        mpv = (xi / self.pitch) * (np.log(a0) + np.log(xi / dEdx_helper.I) + 0.2 - beta * beta - delta)
        return mpv

    def dEdx_pdf(self, dEdx):  # Landau-Vavilov
        beta2 = self.rel_beta2()
        this_xi = self.Landau_xi()
        this_Wmax = self.Get_Wmax()
        this_kappa = this_xi / this_Wmax
        this_meandEdx = self.dEdx_mean()

        par1 = this_xi / self.pitch
        par2 = (0.422784 + beta2 + np.log(this_kappa)) * this_xi / self.pitch + this_meandEdx
        par3 = (dEdx - par2) / par1

        vav = ROOT.Math.VavilovAccurate()
        if this_kappa < 0.01:  # Landau
            this_pdf = ROOT.Math.landau_pdf(par3) / par1
        elif this_kappa > 10:  # Gaussian
            mu = vav.Mean(this_kappa, beta2)
            sigma = np.sqrt(vav.Variance(this_kappa, beta2))
            this_pdf = ROOT.Math.gaussian_pdf(par3, sigma, mu) / par1
        else:  # Vavilov
            this_pdf = vav.Pdf(par3, this_kappa, beta2) / par1
        return this_pdf

    def dEdx_cutoff(self):  # for Landau distribution
        gamma = self.rel_gamma()
        beta2 = self.rel_beta2()
        Wmax = self.Get_Wmax()
        xi = self.Landau_xi()
        kappa = xi / Wmax
        if kappa > 10:  # Gaussian
            return 2 * self.dEdx_mean()
        elif kappa > 0.01:  # Vavilov
            return 100
        else:  # Landau
            lambda_bar = (np.euler_gamma - 1) - beta2 - np.log(kappa)
            lambda_max = 0.51146 + 1.19486 * lambda_bar + (0.465814 + 0.0115374 * lambda_bar) * np.exp(1.17165 + 0.979242 * lambda_bar)
            cutoff = (lambda_max + 1 - np.euler_gamma + beta2 + np.log(kappa)) * (xi / self.pitch) + self.dEdx_mean()
            return cutoff
        
    def dEdx_pdf_gauss_convolved(self, sigma_gaus, dEdx):
        
        this_kappa = self.Landau_xi() / self.Get_Wmax()
        this_a = self.Landau_xi() / self.pitch
        this_mean_dedx = self.dEdx_mean()
        this_b = (0.422784 + self.rel_beta2() + np.log(this_kappa)) * this_a + dEdx
        
        n_points = 500.0
        sc = 5.0 # convolution extends to +-sc Gaussian sigmas                                                                      
        xx = 0.
        yy = 0.
        f_vav = 0.
        sum = 0.
        xlow = 0.
        xupp = 0.
        step = 0
        i = 0.

        xlow = dEdx - sc * sigma_gaus
        xupp = dEdx + sc * sigma_gaus
        step = (xupp-xlow)/n_points
        for i in range(1, int(n_points/2) + 1):
            xx = xlow + (i - 0.5) * step
            yy = (xx - this_b) / this_a
            f_vav = self.dEdx_pdf(yy)
            
            sum += f_vav * ROOT.Math.gaussian_pdf(dEdx, sigma_gaus, xx)

            xx = xupp - (i - 0.5) * step
            yy = (xx - this_b) / this_a
            sum += f_vav * ROOT.Math.gaussian_pdf(dEdx, sigma_gaus, xx)

        invsq2pi = 0.398942280401

        return (step * sum * invsq2pi / sigma_gaus)
