import numpy as np
from scipy import stat




def NONLINEARITY_CORR(fname_nlin, Nsen, verbose=False):

    int_time0 = np.mean(self.int_time[:, Nsen])
    f_nlin = readsav(fname_nlin)

    if abs(f_nlin.iin_-int_time0)>1.0e-5:
        exit('Error [READ_SKS]: Integration time do not match.')

    for iwvl in range(256):
        xx0   = self.spectra_nlin_corr[:,iwvl,Nsen].copy()
        xx    = np.zeros_like(xx0)
        yy    = np.zeros_like(xx0)
        self.spectra_nlin_corr[:,iwvl,Nsen] = -1.0
        logic_xx     = (xx0>-100)
        if verbose:
            print('++++++++++++++++++++++++++++++++++++++++++++++++')
            print('range', f_nlin.mxm_[0,iwvl]*f_nlin.in_[iwvl], f_nlin.mxm_[1,iwvl]*f_nlin.in_[iwvl])
            print('good', logic_xx.sum(), xx0.size)
        xx0[logic_xx] = xx0[logic_xx]/f_nlin.in_[iwvl]

        if (f_nlin.bad_[1,iwvl]<1.0) and (f_nlin.mxm_[0,iwvl]>=1.0e-3):

            #+ data in range (0, minimum)
            yy_e = 0.0
            for ideg in range(f_nlin.gr_):
                yy_e += f_nlin.res2_[ideg,iwvl]*f_nlin.mxm_[0,iwvl]**ideg
            slope = yy_e/f_nlin.mxm_[0,iwvl]
            logic_xx     = (xx0>-100) & (xx0<f_nlin.mxm_[0,iwvl])
            if verbose:
                print('0-min', logic_xx.sum(), xx0.size)
                print('data', xx0[logic_xx])
            xx[xx<0]     = 0.0
            xx[logic_xx] = xx0[logic_xx]
            yy[logic_xx] = xx[logic_xx]*slope

            self.spectra_nlin_corr[logic_xx,iwvl,Nsen] = yy[logic_xx]*f_nlin.in_[iwvl]
            #-

            #+ data in range [minimum, maximum]
            logic_xx     = (xx0>=f_nlin.mxm_[0,iwvl]) & (xx0<=f_nlin.mxm_[1,iwvl])
            xx[logic_xx] = xx0[logic_xx]
            if verbose:
                print('min-max', logic_xx.sum(), xx0.size)
                print('------------------------------------------------')
            for ideg in range(f_nlin.gr_):
                yy[logic_xx] += f_nlin.res2_[ideg, iwvl]*xx[logic_xx]**ideg

            self.spectra_nlin_corr[logic_xx,iwvl,Nsen] = yy[logic_xx]*f_nlin.in_[iwvl]
            #-




if __name__ == '__main__':
    pass
