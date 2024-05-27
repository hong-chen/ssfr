import sys
import warnings
import datetime
import numpy as np



__all__ = ['read_spns']



class read_spns:

    """
    Read data (Diffuse.txt or Total.txt) of SPN-S

    Input:
        fname       : string, file path of the data
        skip_header=: integer, number of header lines to skip, default=3
        info_line=  : integer, number of line that contains wavelength information, default=3
        queit=      : boolen, quiet tag

    Output:
        'read_spns' object that contains:
            self.data['general_info']: python dictionary
            self.data['tmhr']      : numpy array (Ndata,)
            self.data['jday']      : numpy array (Ndata,)
            self.data['wvl']: numpy array (Nwvl,)
            self.data['flux']      : numpy array (Ndata, Nwvl)
    """

    ID = 'CU LASP SPN-S'

    def __init__(self, fname, date_ref=None, skip_header=3, info_line=3, quiet=False):

        # read data
        #/----------------------------------------------------------------------------\#
        with open(fname, 'r') as f:
            lines = f.readlines()

        Nline = len(lines)
        #\----------------------------------------------------------------------------/#


        # set general information
        #/----------------------------------------------------------------------------\#
        self.data = {}
        self.data['general_info'] = {}
        self.data['general_info']['spn_tag'] = 'CU LASP SPN-S'
        self.data['general_info']['fname'] = fname
        #\----------------------------------------------------------------------------/#


        # get wavelength
        #/----------------------------------------------------------------------------\#
        try:
            wvl = np.float_(np.array(lines[info_line-1].strip().split('\t')[1:]))
            self.data['wvl'] = wvl
        except Exception as error:
            print(error)
            msg = '\nError [read_spns]: Cannot interpret the following header line:\n%s' % lines[info_line-1]
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
                    msg = '\nWarning [read_spns]: Cannot interpret the following line data (line #%d):\n%s' % (i+1, lines[i])
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




# for SPN-F
# under development
#/----------------------------------------------------------------------------\#
def read_spn_f_one(fname, skip_header=7):

    with open(fname, 'r') as f:
        lines = f.readlines()

    jday    = np.zeros(len(lines)-skip_header, dtype=np.float64)
    total   = np.zeros_like(jday)
    diffuse = np.zeros_like(jday)

    for i, line in enumerate(lines[skip_header:]):
        line = line.replace('[', '').replace(']', '')
        data = line.split(',')
        dtime = datetime.datetime.strptime(data[0], '%Y%m%d %H:%M:%S.%f')
        jday[i]    = (dtime-datetime.datetime(1, 1, 1)).total_seconds()/86400.0 + 1.0
        total[i]   = data[17]
        diffuse[i] = data[18]

    divide_factor = 10.0*36.4*18.0

    return jday, total/divide_factor, diffuse/divide_factor

class spn_f:

    """
    Read SPN-F data

    Input:
        fnames: SPN-F file paths

    Output:
        'spn_f' object that contains
            self.jday
            self.tmhr
            self.f_total
            self.f_diffuse
    """


    def __init__(self, fnames):

        jday    = np.array([], dtype=np.float64)
        total   = jday.copy()
        diffuse = jday.copy()

        for fname in fnames:
            jday0, total0, diffuse0 = read_spn_f_one(fname)
            jday    = np.append(jday, jday0)
            total   = np.append(total, total0)
            diffuse = np.append(diffuse, diffuse0)

        tmhr = (jday-int(jday[0]))*24.0

        return(tmhr,total,diffuse)

def plot(spn1):

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.scatter(spn1.tmhr, spn1.total, label='Total', color='k')
    ax1.scatter(spn1.tmhr, spn1.diffuse, label='Diffuse', color='b')

    ax2 = ax1.twinx()
    ax2.scatter(spn1.tmhr, spn1.sun, color='r', label='Sun?')

    ax1.set_title('SPN1 %s' % os.path.basename(spn1.fname))
    ax1.set_ylabel('Flux [$\mathrm{W m^{-2}}$]')
    ax1.set_xlabel('Time [Hour]')

    ax2.set_ylim((-1, 2))
    ax2.set_ylabel('Sun?')
    ax1.legend(loc='upper right', fontsize=18)
    ax2.legend(loc='upper left', fontsize=18)
    plt.savefig(spn1.fname.replace('SKS', 'png'))
    # ---------------------------------------------------------------------

class spn_f_sks:

    """
    Read SPN-F data with extension of .SKS (e.g., Norway data)

    Input:
        fname: string, file path of the SPN-F data

    Output:
        'spn_f_sks' object that contains:
            self.jday      = jday
            self.tmhr      = tmhr
            self.f_total   = total
            self.f_diffuse = diffuse
            self.f_direct  = total-diffuse
            self.f_sun     = sun

    """

    def __init__(self, fname, skip_header=7):

        self.fname = fname

        with open(fname, 'r', encoding='latin-1') as f:
            lines = f.readlines()

        Nline = len(lines)
        Ndata = (Nline-skip_header) // 2
        Nleft = (Nline-skip_header)  % 2

        if Nleft != 0:
            skip_header += 1
            Ndata -= 1

            # exit('Error [READ_SPN1]: %s has invalid number of data records.' % fname)

        jday    = np.zeros(Ndata, dtype=np.float64)
        total   = np.zeros(Ndata, dtype=np.float64)
        diffuse = np.zeros(Ndata, dtype=np.float64)
        sun     = np.zeros(Ndata, dtype=np.int32)

        for i in range(Ndata):

            # extract the data
            # +
            index_data = skip_header+2*i
            line_data  = lines[index_data].replace('S', '').replace(' ', '')

            total0, diffuse0, sun0 = line_data.split(',')
            total[i]   = float(total0)
            diffuse[i] = float(diffuse0)
            sun[i]     = int(sun0)
            # -

            # extract the time
            # +
            index_time = (skip_header+1)+2*i

            line_time            = lines[index_time].replace('+', '').replace('?', '').replace('¯', '')
            date, time, _        = line_time.split()
            month, day, year     = date.split('/')
            hour, minute, second = time.split(':')
            year   = int(year)
            month  = int(month)
            day    = int(day)
            hour   = int(hour)
            minute = int(minute)
            second = int(second)

            jday[i] = (datetime.datetime(year, month, day, hour, minute, second) - datetime.datetime(year-1, 12, 31)).total_seconds() / 86400.0
            if 'PM' in line_time:
                jday[i] += 0.5

            # -

        tmhr = (jday-int(jday[0])) * 24.0

        self.jday      = jday
        self.tmhr      = tmhr
        self.f_total   = total
        self.f_diffuse = diffuse
        self.f_direct  = total-diffuse
        self.f_sun     = sun
#\----------------------------------------------------------------------------/#




if __name__ == '__main__':

    pass
