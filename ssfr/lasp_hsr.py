import sys
import warnings
import datetime
import numpy as np

import ssfr.util



__all__ = ['read_hsr1']



class read_hsr1:

    """
    Read data (Diffuse.txt or Total.txt) of HSR1

    Input:
        fname       : string, file path of the data
        skip_header=: integer, number of header lines to skip, default=3
        info_line=  : integer, number of line that contains wavelength information, default=3
        queit=      : boolen, quiet tag

    Output:
        'read_hsr1' object that contains:
            self.data['general_info']: python dictionary
            self.data['tmhr']      : numpy array (Ndata,)
            self.data['jday']      : numpy array (Ndata,)
            self.data['wvl']: numpy array (Nwvl,)
            self.data['flux']      : numpy array (Ndata, Nwvl)
    """

    ID = 'CU LASP HSR1'

    def __init__(self, fname, date_ref=None, skip_header=3, info_line=3, quiet=False):

        # read data
        #/----------------------------------------------------------------------------\#
        with open(fname, 'r') as f:
            msg = '\nMessage [read_hsr1]: Processing CU-LASP HSR1 file:\n    reading <%s> ...' % (fname)
            print(msg)
            lines = f.readlines()

        Nline = len(lines)
        #\----------------------------------------------------------------------------/#


        # set general information
        #/----------------------------------------------------------------------------\#
        self.data = {}
        self.data['general_info'] = {}
        self.data['general_info']['hsr_tag'] = 'CU LASP HSR1'
        self.data['general_info']['fname'] = fname
        #\----------------------------------------------------------------------------/#


        # get wavelength
        #/----------------------------------------------------------------------------\#
        try:
            wvl = np.float_(np.array(lines[info_line-1].strip().split('\t')[1:]))
            self.data['wvl'] = wvl
        except Exception as error:
            print(error)
            msg = '\nError [read_hsr1]: Cannot interpret the following header line:\n%s' % lines[info_line-1]
            raise OSError(msg)
        #\----------------------------------------------------------------------------/#


        # get julian day (with reference to date 0001/01/01) and fluxes
        #/----------------------------------------------------------------------------\#
        jday = np.zeros( Nline-skip_header           , dtype=np.float64); jday[...] = np.nan
        flux = np.zeros((Nline-skip_header, wvl.size), dtype=np.float64); flux[...] = np.nan

        for i in range(skip_header, Nline):

            try:
                line = lines[i].strip()
                data = line.split('\t')
                dtime_s = ' '.join(data[0].split(' ')[:2])
                dtime                  = datetime.datetime.strptime(dtime_s, '%Y-%m-%d %H:%M:%S')
                jday[i-skip_header]    = (dtime  - datetime.datetime(1, 1, 1)).total_seconds() / 86400.0 + 1.0
                flux[i-skip_header, :] = np.float_(np.array(data[1:]))
            except Exception as error:
                print(error)
                if not quiet:
                    msg = '\nWarning [read_hsr1]: Cannot interpret the following line data (line #%d):\n%s' % (i+1, lines[i])
                    warnings.warn(msg)

        self.data['jday'] = jday
        self.data['flux'] = flux
        #\----------------------------------------------------------------------------/#


        # get time in hour (tmhr)
        #/----------------------------------------------------------------------------\#
        if date_ref is None:
            # find the most frequent integer (non-NaN) of jday and calculate time in hour
            tmhr = (jday-np.bincount(np.int_(jday[np.logical_not(np.isnan(jday))])).argmax()) * 24.0
        else:
            jday_ref  = (date_ref - datetime.datetime(1, 1, 1)).total_seconds()/86400.0 + 1.0
            tmhr = (sjday-jday_ref) * 24.0

        self.data['tmhr'] = tmhr
        #\----------------------------------------------------------------------------/#

        dtime_s0 = ssfr.util.jday_to_dtime(self.data['jday'][0]).strftime('%Y-%m-%d %H:%M:%S')
        dtime_e0 = ssfr.util.jday_to_dtime(self.data['jday'][-1]).strftime('%Y-%m-%d %H:%M:%S')
        msg = '\nMessage [read_hsr1]: Data processing complete (%s to %s).' % (dtime_s0, dtime_e0)
        print(msg)



if __name__ == '__main__':

    pass
