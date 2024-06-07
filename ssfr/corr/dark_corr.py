import sys
import warnings
import numpy as np
from scipy import stats


__all__ = ['dark_corr']



def dark_corr(
        x0,
        shutter0,
        data0,
        mode='interp',
        dark_extend=1,
        light_extend=1,
        light_threshold=10,
        dark_threshold=5,
        shutter_mode={'open':0, 'close':1},
        fill_value=np.nan,
        temp_threshold=25.0,
        verbose=False
        ):

    # size check
    #/----------------------------------------------------------------------------\#
    if x0.size != shutter0.size:
        msg = '\nError [dark_corr]: <shuttr0.size> does not match <x0.size>.'
        raise OSError(msg)
    #\----------------------------------------------------------------------------/#


    # dimension check
    #/----------------------------------------------------------------------------\#
    if data0.ndim == 1:
        Nx = data0.size
        if Nx != x0.size:
            msg = '\nError [dark_corr]: <data0.size> does not match <x0.size>.'
            raise OSError(msg)

    elif data0.ndim == 2:
        Nx, Ny = data0.shape
        if Nx == x0.size:
            swapAxis = False
        elif Ny == x0.size:
            data0 = data0.T
            Nx, Ny = data0.shape
            swapAxis = True
        else:
            msg = '\nError [dark_corr]: None of the axis in <data0.ndim> match <x0.size>.'
            raise OSError(msg)

    else:
        msg = '\nError [dark_corr]: Do not support <data0.ndim> greater than 2.'
        raise OSError(msg)
    #\----------------------------------------------------------------------------/#


    # make a copy of the data so the original data won't get overwritten in memory
    #/----------------------------------------------------------------------------\#
    x       = x0.copy()
    shutter = shutter0.copy()
    data    = data0.copy()
    #\----------------------------------------------------------------------------/#


    # append a different status at the begin and end to force cycle break
    #/----------------------------------------------------------------------------\#
    if shutter[0] == shutter_mode['open']:
        shutter_append = np.append(shutter_mode['close'], shutter)
    elif shutter[0] == shutter_mode['close']:
        shutter_append = np.append(shutter_mode['open'], shutter)

    if shutter[-1] == shutter_mode['open']:
        shutter_append = np.append(shutter_append, shutter_mode['close'])
    elif shutter[-1] == shutter_mode['close']:
        shutter_append = np.append(shutter_append, shutter_mode['open'])

    shutter_diff = shutter_append[1:]-shutter_append[:-1]
    #\----------------------------------------------------------------------------/#


    # identify dark and light cycles
    #/----------------------------------------------------------------------------\#
    break_indices = np.where(shutter_diff!=0)[0]

    logic_light    = np.repeat(False, shutter.size)
    logic_dark     = np.repeat(False, shutter.size)

    circle_range   = []
    circle_tag     = []

    for i in range(break_indices.size-1):

        index_l0 = break_indices[i]
        index_r0 = break_indices[i+1]

        shutter0     = shutter[index_l0:index_r0]
        shutter0_uni = np.unique(shutter0)

        if shutter0_uni.size != 1:
            msg = '\nError [dark_corr]: The break indices are wrong.'
            raise ValueError(msg)

        if shutter0_uni[0] == shutter_mode['open']:
            index_l = max([0, index_l0+light_extend])
            index_r = min([index_r0-light_extend, shutter.size])
            if (index_r-index_l) > light_threshold:
                circle_range.append([index_l, index_r])
                circle_tag.append(shutter_mode['open'])
                logic_light[index_l:index_r] = True

        elif shutter0_uni[0] == shutter_mode['close']:
            index_l = max([0, index_l0+dark_extend])
            index_r = min([index_r0-dark_extend, shutter.size])
            if (index_r-index_l) > dark_threshold:
                circle_range.append([index_l, index_r])
                circle_tag.append(shutter_mode['close'])
                logic_dark[index_l:index_r] = True
    #\----------------------------------------------------------------------------/#


    # shutter
    #/----------------------------------------------------------------------------\#
    if 'exclude' not in shutter_mode.keys():
        shutter_mode['excluded'] = 10
    logic_excluded = np.logical_not(logic_dark|logic_light)
    shutter[logic_excluded] = shutter_mode['excluded']
    #\----------------------------------------------------------------------------/#


    # perform dark correction
    #/----------------------------------------------------------------------------\#
    data_corr      = np.zeros_like(data)
    data_corr[...] = fill_value

    # dark correction mode, if only one mode is detected, return average
    #/--------------------------------------------------------------\#
    mode = mode.lower()
    shutter_modes = np.unique(shutter)
    if (shutter_modes.size == 1):
        msg = '\nWarning [dark_corr]: Only one light/dark cycle is detected, returning average ...'
        warnings.warn(msg)
        return np.mean(data[~logic_excluded, ...], axis=0)
    #\--------------------------------------------------------------/#

    if mode == 'mean':

        dark_mean = np.mean(data[logic_dark, ...], axis=0)
        data_corr[logic_light, ...] = data[logic_light, ...] - np.tile(dark_mean, logic_light.sum()).reshape(-1, Ny)

    elif mode == 'interp':

        Ncircle = len(circle_tag)

        for i in range(Ncircle):

            ctag   = circle_tag[i]
            crange = circle_range[i]

            if ctag == shutter_mode['open']:

                Nl = i-1
                while Nl >= 0 and circle_tag[Nl] != shutter_mode['close']:
                    Nl -= 1

                Nr = i+1
                while Nr <= (Ncircle-1) and circle_tag[Nr] != shutter_mode['close']:
                    Nr += 1

                if Nl<0:
                    msg = '\nWarnings [dark_corr]: Found light cycle at the very beginning, use average darks from the next available dark cycle ...'
                    warnings.warn(msg)
                    dark_mean = np.mean(data[circle_range[Nr][0]:circle_range[Nr][1], ...], axis=0)
                    data_corr[crange[0]:crange[1], ...] = data[crange[0]:crange[1], ...] - np.tile(dark_mean, crange[1]-crange[0]).reshape(-1, Ny)

                    if 'interp_begin' not in shutter_mode.keys():
                        shutter_mode['interp_begin'] = -10
                    shutter[crange[0]:crange[1]] = shutter_mode['interp_begin']

                elif Nr>(Ncircle-1):
                    msg = '\nWarnings [dark_corr]: Found light cycle at the very end, use average darks from the previous available dark cycle ...'
                    warnings.warn(msg)
                    dark_mean = np.mean(data[circle_range[Nl][0]:circle_range[Nl][1], ...], axis=0)
                    data_corr[crange[0]:crange[1], ...] = data[crange[0]:crange[1], ...] - np.tile(dark_mean, crange[1]-crange[0]).reshape(-1, Ny)

                    if 'interp_end' not in shutter_mode.keys():
                        shutter_mode['interp_end'] = -11
                    shutter[crange[0]:crange[1]] = shutter_mode['interp_end']
                else:
                    interp_x = np.append(x[circle_range[Nl][0]:circle_range[Nl][1]], x[circle_range[Nr][0]:circle_range[Nr][1]])
                    target_x = x[crange[0]:crange[1]]
                    for iChan in range(Ny):
                        interp_y = np.append(data[circle_range[Nl][0]:circle_range[Nl][1], iChan], data[circle_range[Nr][0]:circle_range[Nr][1], iChan])
                        slope, intercept, r_value, p_value, std_err = stats.linregress(interp_x, interp_y)
                        dark_offset = target_x*slope + intercept
                        data_corr[crange[0]:crange[1], iChan] = data[crange[0]:crange[1], iChan] - dark_offset

    elif mode == 'temp':

        msg = '\nWarnings [dark_corr]: Performing temperature dependent correction, please make sure input <x> is a temperature variable ...'
        warnings.warn(msg)

        logic_fit = logic_dark & (x>temp_threshold)

        x_fit = x[logic_fit]
        for iChan in range(Ny):
            y_fit = data[logic_fit, iChan]
            coef  = np.polyfit(x_fit, y_fit, 5)
            data_corr[logic_light, iChan] = data[logic_light, iChan] - np.polyval(coef, x[logic_light])

    else:
        msg = '\nError [dark_corr]: <mode=%s> has not been implemented yet.' % mode
        raise OSError(msg)
    #\----------------------------------------------------------------------------/#

    if swapAxis:
        return shutter, data_corr.T
    else:
        return shutter, data_corr



def dark_corr_old(
        x0,
        shutter0,
        data0,
        mode='interp',
        dark_extend=2,
        light_extend=2,
        light_threshold=10,
        dark_threshold=5,
        shutter_mode={'open':0, 'close':1},
        fill_value=np.nan,
        verbose=False):

    x       = x0.copy()
    shutter = shutter0.copy()
    data    = data0.copy()

    data_corr      = np.zeros_like(data)
    data_corr[...] = fill_value

    mode          = mode.lower()
    shutter_modes = np.unique(shutter)
    if (shutter_modes.size == 1) and (mode != 'mean'):
        print('Warning [dark_corr]: Only one light/dark cycle is detected, change \'mode="%s"\' to \'mode="mean"\'.' % mode)
        mode = 'mean'


    if data.ndim == 1:

        pass

    elif data.ndim == 2:

        Nx, Ny = data.shape

        print(Nx, Ny)

    exit()


    # only dark or light cycle present
    if np.unique(shutter).size == 1:

        if mode.lower() != 'mean':
            print('Warning [dark_corr]: only one light/dark cycle is detected, \'%s\' is not supported, return fill value.' % mode)
            return shutter, spectra_dark_corr
        else:
            if np.unique(shutter)[0] == 0:
                if verbose:
                    print('Warning [dark_corr]: only one light cycle is detected.')
                mean = np.mean(spectra[light_extend:-light_extend, :], axis=0)
                spectra_dark_corr = np.tile(mean, spectra.shape[0]).reshape(spectra.shape)
                return shutter, spectra_dark_corr
            elif np.unique(shutter)[0] == 1:
                if verbose:
                    print('Warning [dark_corr]: only one dark cycle is detected.')
                mean = np.mean(spectra[dark_extend:-dark_extend, :], axis=0)
                spectra_dark_corr = np.tile(mean, spectra.shape[0]).reshape(spectra.shape)
                return shutter, spectra_dark_corr
            else:
                sys.exit('Error   [dark_corr]: cannot interpret shutter status.')

    # both dark and light cycles present
    else:

        dark_offset            = np.zeros_like(spectra)
        dark_std               = np.zeros_like(spectra)
        dark_offset[...]       = fill_value
        dark_std[...]          = fill_value

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
            sys.exit('Error   [dark_corr]: darkL and darkR do not match.')
        # ??????????????????????????????????????????????????????????????????????????????

        if darkL.size != darkR.size:
            sys.exit('Error   [dark_corr]: the number of dark cycles is incorrect.')

        if mode.lower() == 'dark_interpolate':

            shutter[:darkL[0]+dark_extend] = fill_value  # omit the data before the first dark cycle

            for i in range(darkL.size-1):

                if darkR[i] < darkL[i]:
                    sys.exit('Error   [dark_corr]: darkR[%d]=%d is smaller than darkL[%d]=%d.' % (i,darkR[i],i,darkL[i]))

                darkLL = darkL[i]   + dark_extend
                darkLR = darkR[i]   - dark_extend
                darkRL = darkL[i+1] + dark_extend
                darkRR = darkR[i+1] - dark_extend
                lightL = darkR[i]   + light_extend
                lightR = darkL[i+1] - light_extend

                shutter[darkL[i]:darkLL] = fill_value
                shutter[darkLR:darkR[i]] = fill_value
                shutter[darkR[i]:lightL] = fill_value
                shutter[lightR:darkL[i+1]] = fill_value
                shutter[darkL[i+1]:darkRL] = fill_value
                shutter[darkRR:darkR[i+1]] = fill_value

                if lightR-lightL>light_threshold and darkLR-darkLL>dark_threshold and darkRR-darkRL>dark_threshold:

                    interp_x = np.append(tmhr[darkLL:darkLR], tmhr[darkRL:darkRR])
                    target_x = tmhr[darkL[i]:darkL[i+1]]

                    for iChan in range(Nchannel):
                        interp_y = np.append(spectra[darkLL:darkLR, iChan], spectra[darkRL:darkRR, iChan])
                        slope, intercept, r_value, p_value, std_err  = stats.linregress(interp_x, interp_y)
                        dark_offset[darkL[i]:darkL[i+1], iChan] = target_x*slope + intercept
                        dark_std[darkL[i]:darkL[i+1], iChan]    = np.std(interp_y)
                        spectra_dark_corr[darkL[i]:darkL[i+1], iChan] = spectra[darkL[i]:darkL[i+1], iChan] - dark_offset[darkL[i]:darkL[i+1], iChan]

                else:
                    shutter[darkL[i]:darkR[i+1]] = fill_value

            shutter[darkRR:] = fill_value  # omit the data after the last dark cycle
            spectra_dark_corr[shutter==fill_value, :] = fill_value

            return shutter, spectra_dark_corr

        else:
            sys.exit('Error   [dark_corr]: \'%s\' has not been implemented yet.' % mode)



if __name__ == "__main__":

    pass
