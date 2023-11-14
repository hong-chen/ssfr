import os
import sys
import glob
import datetime
import numpy as np

import ssfr

# adapted from IDL code
# hg=[296.73,302.15,312.57,313.17,334.15,365.02,365.48,366.33,404.66,407.78,433.92,434.75,435.48,491.6,546.07,576.96,579.07,1014]
lamps_idl = {
        'hg': np.array([
            296.73, 302.15, 312.57, 313.17,
            334.15, 365.02, 365.48, 366.33,
            404.66, 407.78, 433.92, 434.75,
            435.48, 491.60, 546.07, 576.96,
            579.07, 1014.0
            ])
        }

# asterisk wavelengths (no neighbouring lines within 1.0 nm) from lamp manual
lamps = {
        'hg': np.array([
            296.7283, 334.1484, 404.6565, 407.7837,
            435.8335, 546.0750, 576.9610, 579.0670,
            1013.979, 1128.741, 1357.021, 1367.351,
            1395.055, 1529.597
            ]),
        'kr': np.array([
            450.235,  605.611,  758.741,  760.154,
            805.950,  828.105,  850.887,  892.869,
            985.624, 1022.146, 1145.748, 1181.938,
           1363.422, 1442.679, 1473.444, 1523.962,
           1678.513, 1693.581, 1816.732, 2190.251
            ])
        }


def load_wvl_coef(fname='%s/wvl_coef.dat' % ssfr.common.fdir_data):

    with open(fname, 'r') as f:
        lines = f.readlines()

    coefs = {}
    for line_ in lines:
        line = line_.strip().replace(' ', '').replace('\n', '')
        if line[0] != '#':
            data  = line.split(',')
            vname = data[0]
            coef  = np.array([float(data0) for data0 in data[1:]])
            if vname not in coefs.keys():
                coefs[vname] = coef

    return coefs


def cal_wvl(coef, Nchan=256):

    xchan = np.arange(Nchan, dtype=np.float64)

    wvl = np.zeros_like(xchan)
    for i, coef0 in enumerate(coef):
        wvl += coef0 * xchan**i

    return wvl


def poly_fit():

    pass


if __name__ == '__main__':

    coef = load_wvl_coef(fname='%s/wvl_coef.dat' % ssfr.common.fdir_data)

    print(cal_wvl(coef['lasp|ssfr-a|zen|si']))

    pass
