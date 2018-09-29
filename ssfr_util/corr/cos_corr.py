import h5py
import numpy as np
from scipy import interpolate
from scipy.io import readsav

from ssfr_util.util import IF_FILE_EXISTS


def LOAD_COSINE_RESPONSE(fname):

    IF_FILE_EXISTS(fname, exitTag=True)

    try:
        f    = h5py.File(fname, 'r')
        mu   = f['mu'][...]
        wvl  = f['wvl'][...]
        resp = f['resp'][...]
        f.close()
        return mu, wvl, resp
    except OSError:
        print("Error [LOAD_COSINE_RESPONSE]: cannot open '{fname}'.".format(fname=fname))
    except KeyError:
        print("Error [LOAD_COSINE_RESPONSE]: cannot read ('wvl', 'resp', 'mu') from '{fname}'.".format(fname=fname))



class SSFR_COS_CORR:

    def __init__(self, date, pit_offset=0.0, rol_offset=0.0, tag=None,
                 fname_zen = '/Users/hoch4240/Chen/work/01_ARISE/cos_resp/lamp/ZENITH.idl',
                 fname_nad = '/Users/hoch4240/Chen/work/01_ARISE/cos_resp/lamp/NADIR.idl'):

        f_alp = LOAD_ALP_H5(date, tag=tag)
        f_bbr = LOAD_BBR_H5(date, tag=tag)
        f_ssfr= LOAD_SSFR_01_H5(date, tag=tag)

        iza , iaa  = PRH2ZA(f_alp.pit-f_alp.pit_m-pit_offset, f_alp.rol-f_alp.rol_m-rol_offset, f_alp.hed)
        iza0, iaa0 = PRH2ZA(np.repeat(0.0, f_alp.pit.size), np.repeat(0.0, f_alp.rol.size), f_alp.hed)
        self.sza, self.saa, self.iza, self.iaa = f_alp.sza, f_alp.saa, iza, iaa

        dc       = MUSLOPE(f_alp.sza, f_alp.saa, iza, iaa)
        dc0      = MUSLOPE(f_alp.sza, f_alp.saa, iza0, iaa0)
        self.dc, self.dc0 = dc, dc0

        indice  = np.int_(np.round(dc , decimals=3)*1000.0)
        indice[indice>1000] = 1000
        indice[indice<0]    = 0

        indice0 = np.int_(np.round(dc0, decimals=3)*1000.0)
        indice0[indice0>1000] = 1000
        indice0[indice0<0]    = 0

        # zenith cosine response
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        f_cos = readsav(fname_zen)
        wvl   = f_cos.wlf
        resp  = f_cos.dr
        mu_i  = f_cos.mu
        wvl0   = 666.0
        index0 = np.argmin(np.abs(wvl-wvl0))
        resp0  = mu_i/resp[:, index0]
        resp0[resp0<0.0000001] = 0.0000001

        # # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # fig = plt.figure(figsize=(8, 5))
        # ax1 = fig.add_subplot(111)
        # ax1.set_xlim((0.0, 1.0))
        # ax1.set_ylim((0.0, 1.2))
        # ax1.plot(mu_i, resp0, label='Cosine Reponse')
        # ax1.plot([0.0, 1.0], [0.0, 1.0], label='1:1 Line', color='gray', ls='--')
        # ax1.set_xlabel('cos($\\theta$)')
        # ax1.set_ylabel('Response Function')
        # # ax1.set_xlim(())
        # # ax1.set_ylim(())
        # ax1.legend(loc='upper left', framealpha=0.4)
        # plt.savefig('/Users/hoch4240/Chen/mygit/slides/data/comps/cos_resp_zen.png')
        # plt.show()
        # exit()
        # # ---------------------------------------------------------------------

        self.dc_cos = resp0[indice]

        cos_corr_factor_mean= 0.5 / np.trapz(resp0, x=mu_i)
        # cos_corr_factor_dn  = (f_bbr.diff_ratio*cos_corr_factor_mean + (1.0-f_bbr.diff_ratio)*(np.cos(np.deg2rad(f_alp.sza)) / resp0[indice]))
        # cos_corr_factor_dn0 = (f_bbr.diff_ratio*cos_corr_factor_mean + (1.0-f_bbr.diff_ratio)*(np.cos(np.deg2rad(f_alp.sza)) / resp0[indice0]))
        cos_corr_factor_dn  = (f_ssfr.diff_ratio_zen*cos_corr_factor_mean + (1.0-f_ssfr.diff_ratio_zen)*(np.cos(np.deg2rad(f_alp.sza)) / resp0[indice]))
        cos_corr_factor_dn0 = (f_ssfr.diff_ratio_zen*cos_corr_factor_mean + (1.0-f_ssfr.diff_ratio_zen)*(np.cos(np.deg2rad(f_alp.sza)) / resp0[indice0]))

        self.cos_corr_factor_dn, self.cos_corr_factor_dn0 = cos_corr_factor_dn, cos_corr_factor_dn0
        # -----------------------------------------------------------------------------------------------------

        # nadir cosine response
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        f_cos = readsav(fname_nad)
        # wvl   = f_cos.wlf
        # resp  = f_cos.dr
        # mu_i  = f_cos.mu
        # wvl0   = 666.0
        # index0 = np.argmin(np.abs(wvl-wvl0))
        # resp0  = mu_i/resp[:, index0]
        # resp0[resp0<0.0000001] = 0.0000001

        # cos_corr_factor_up  = 0.5 / np.trapz(resp0, x=mu_i)

        # self.cos_corr_factor_up = cos_corr_factor_up
        self.cos_corr_factor_up = np.interp(f_ssfr.wvl_nad, f_cos.wlf, f_cos.df) * 1.0
        # -----------------------------------------------------------------------------------------------------

        self.logic = (self.cos_corr_factor_dn>0.0) & (self.dc>0.0) & (self.dc<1.0) & f_alp.logic


if __name__ == '__main__':
    pass
