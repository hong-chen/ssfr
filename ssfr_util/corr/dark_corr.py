import numpy as np
from scipy import stats




def DARK_CORRECTION(tmhr0, shutter0, spectra0, mode="dark_interpolate", darkExtend=2, lightExtend=2, lightThr=10, darkThr=5, fillValue=-99999, verbose=False):

    tmhr              = tmhr0.copy()
    shutter           = shutter0.copy()
    spectra           = spectra0.copy()
    Nrecord, Nchannel = spectra.shape

    spectra_dark_corr      = np.zeros_like(spectra)
    spectra_dark_corr[...] = fillValue

    # only dark or light cycle present
    if np.unique(shutter).size == 1:

        if mode.lower() != 'mean':
            print('Warning [DARK_CORRECTION]: only one light/dark cycle is detected, \'%s\' is not supported, return fill value.' % mode)
            return spectra_dark_corr
        else:
            if np.unique(shutter)[0] == 0:
                if verbose:
                    print('Warning [DARK_CORRECTION]: only one light cycle is detected.')
                mean = np.mean(spectra[lightExtend:-lightExtend, :], axis=0)
                spectra_dark_corr = np.tile(mean, spectra.shape[0]).reshape(spectra.shape)
                return spectra_dark_corr
            elif np.unique(shutter)[0] == 1:
                if verbose:
                    print('Warning [DARK_CORRECTION]: only one dark cycle is detected.')
                mean = np.mean(spectra[darkExtend:-darkExtend, :], axis=0)
                spectra_dark_corr = np.tile(mean, spectra.shape[0]).reshape(spectra.shape)
                return spectra_dark_corr
            else:
                print('Exit [DARK_CORRECTION]: cannot interpret shutter status.')

    # both dark and light cycles present
    else:

        dark_offset            = np.zeros_like(spectra)
        dark_std               = np.zeros_like(spectra)
        dark_offset[...]       = fillValue
        dark_std[...]          = fillValue

        if shutter[0] == 0:
            darkL = np.array([], dtype=np.int32)
            darkR = np.array([0], dtype=np.int32)
        else:
            darkR = np.array([], dtype=np.int32)
            darkL = np.array([0], dtype=np.int32)

        darkL0 = np.squeeze(np.argwhere((shutter[1:]-shutter[:-1]) ==  1)) + 1
        darkL  = np.hstack((darkL, darkL0))

        darkR0 = np.squeeze(np.argwhere((shutter[1:]-shutter[:-1]) == -1)) + 1
        darkR  = np.hstack((darkR, darkR0))

        if shutter[-1] == 0:
            darkL = np.hstack((darkL, shutter.size))
        else:
            darkR = np.hstack((darkR, shutter.size))

        # ??????????????????????????????????????????????????????????????????????????????
        # this part might need more work
        if darkL.size-darkR.size==0:
            if darkL[0]>darkR[0] and darkL[-1]>darkR[-1]:
                darkL = darkL[:-1]
                darkR = darkR[1:]
        elif darkL.size-darkR.size==1:
            if darkL[0]>darkR[0] and darkL[-1]<darkR[-1]:
                darkL = darkL[1:]
            elif darkL[0]<darkR[0] and darkL[-1]>darkR[-1]:
                darkL = darkL[:-1]
        elif darkR.size-darkL.size==1:
            if darkL[0]>darkR[0] and darkL[-1]<darkR[-1]:
                darkR = darkR[1:]
            elif darkL[0]<darkR[0] and darkL[-1]>darkR[-1]:
                darkR = darkR[:-1]
        else:
            exit('Error [DARK_CORRECTION]: darkL and darkR do not match.')
        # ??????????????????????????????????????????????????????????????????????????????

        if darkL.size != darkR.size:
            exit('Error [DARK_CORRECTION]: the number of dark cycles is incorrect.')

        if mode.lower() == 'dark_interpolate':

            shutter[:darkL[0]+darkExtend] = fillValue  # omit the data before the first dark cycle

            for i in range(darkL.size-1):

                if darkR[i] < darkL[i]:
                    exit('Error [DARK_CORRECTION]: darkR[%d]=%d is smaller than darkL[%d]=%d.' % (i,darkR[i],i,darkL[i]))

                darkLL = darkL[i]   + darkExtend
                darkLR = darkR[i]   - darkExtend
                darkRL = darkL[i+1] + darkExtend
                darkRR = darkR[i+1] - darkExtend
                lightL = darkR[i]   + lightExtend
                lightR = darkL[i+1] - lightExtend

                shutter[darkL[i]:darkLL] = fillValue
                shutter[darkLR:darkR[i]] = fillValue
                shutter[darkR[i]:lightL] = fillValue
                shutter[lightR:darkL[i+1]] = fillValue
                shutter[darkL[i+1]:darkRL] = fillValue
                shutter[darkRR:darkR[i+1]] = fillValue

                if lightR-lightL>lightThr and darkLR-darkLL>darkThr and darkRR-darkRL>darkThr:

                    interp_x = np.append(tmhr[darkLL:darkLR], tmhr[darkRL:darkRR])
                    target_x = tmhr[darkL[i]:darkL[i+1]]

                    for iChan in range(Nchannel):
                        interp_y = np.append(spectra[darkLL:darkLR, iChan], spectra[darkRL:darkRR, iChan])
                        slope, intercept, r_value, p_value, std_err  = stats.linregress(interp_x, interp_y)
                        dark_offset[darkL[i]:darkL[i+1], iChan] = target_x*slope + intercept
                        dark_std[darkL[i]:darkL[i+1], iChan]    = np.std(interp_y)
                        spectra_dark_corr[darkL[i]:darkL[i+1], iChan] = spectra[darkL[i]:darkL[i+1], iChan] - dark_offset[darkL[i]:darkL[i+1], iChan]

                else:
                    shutter[darkL[i]:darkR[i+1]] = fillValue

            shutter[darkRR:] = fillValue  # omit the data after the last dark cycle
            spectra_dark_corr[shutter==fillValue, :] = fillValue

            return spectra_dark_corr

        else:
            exit('Error [DARK_CORRECTION]: \'%s\' has not been implemented yet.' % mode)





if __name__ == "__main__":

    pass
