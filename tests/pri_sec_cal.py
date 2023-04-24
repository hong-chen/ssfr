
def PLOT():
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    rcParams['font.size'] = 18
    fig = plt.figure(figsize=(7, 6))
    ax1 = fig.add_subplot(111)
    ax1.scatter(xChan, self.wvl_zen_si, label='Silicon', c='red')
    ax1.scatter(xChan, self.wvl_zen_in, label='InGaAs', c='blue')
    # ax1.scatter(xChan, self.wvl_nad_si, label='Silicon', c='red')
    # ax1.scatter(xChan, self.wvl_nad_in, label='InGaAs', c='blue')
    ax1.set_xlim((0, 255))
    ax1.set_ylim((250, 2250))
    ax1.set_xlabel('Channel Number')
    ax1.set_ylabel('Wavelength [nm]')
    # ax1.set_title('Nadir')
    ax1.set_title('Zenith')
    ax1.legend(loc='upper right', framealpha=0.4, fontsize=18)
    plt.savefig('wvl_zenith.png')
    plt.show()
    exit()
    # ---------------------------------------------------------------------

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(5, 3))
    ax1 = fig.add_subplot(111)
    # ax1.scatter(self.wvl_zen_si, lampStd_zen_si)
    # ax1.scatter(self.wvl_zen_in, lampStd_zen_in)
    ax1.scatter(self.wvl_nad_si, lampStd_nad_si)
    # ax1.scatter(self.wvl_nad_in, lampStd_nad_in)
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')
    # ax1.set_xlim((200, 2600))
    # ax1.xaxis.set_major_locator(FixedLocator(np.arange(200, 2601, 400)))
    # ax1.set_ylim((0.0, 0.25))
    # ax1.set_title('Zenith Silicon')
    # ax1.set_title('Zenith InGaAs')
    ax1.set_title('Nadir Silicon')
    # ax1.set_title('Nadir InGaAs')
    # ax1.legend(loc='upper right', fontsize=10, framealpha=0.4)
    # plt.savefig('std_zen_si.png')
    # plt.savefig('std_zen_in.png')
    plt.savefig('std_nad_si.png')
    # plt.savefig('std_nad_in.png')
    plt.show()
    exit()
    # ---------------------------------------------------------------------

def PLOT_PRIMARY_RESPONSE_20180711():

    from ssfr_config import config_20180711_a1, config_20180711_a2, config_20180711_a3, config_20180711_b1, config_20180711_b2, config_20180711_b3

    markers     = ['D', '*']
    markersizes = [10, 4]
    linestyles = ['-', '--']
    colors     = ['red', 'blue', 'green']
    linewidths = [1.0, 1.0]
    alphas     = [1.0, 1.0]

    cal_a1 = CALIBRATION_CU_SSFR(config_20180711_a1)
    cal_a2 = CALIBRATION_CU_SSFR(config_20180711_a2)
    cal_a3 = CALIBRATION_CU_SSFR(config_20180711_a3)

    cals_a = [cal_a1, cal_a2, cal_a3]

    cal_b1 = CALIBRATION_CU_SSFR(config_20180711_b1)
    cal_b2 = CALIBRATION_CU_SSFR(config_20180711_b2)
    cal_b3 = CALIBRATION_CU_SSFR(config_20180711_b3)

    cals_b = [cal_b1, cal_b2, cal_b3]

    cals = [cals_a, cals_b]
    labels  = [['SSIM1 S45-90/I250-375', 'SSIM1 S45-45/I250-250', 'SSIM1 S90-90/I375-375'], ['10862 S45-90/I250-375', '10862 S45-45/I250-250', '10862 S90-90/I375-375']]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(111)
    for i, cals0 in enumerate(cals):
        for j, cal0 in enumerate(cals0):
            label0 = labels[i][j]

            # if 'SSIM1' in label0:
            intTimes_si = list(cal0.primary_response_zen_si.keys())
            intTimes_in = list(cal0.primary_response_zen_in.keys())

            for k in range(len(intTimes_si)):

                label = '%s (S%dI%d)' % (label0, intTimes_si[k], intTimes_in[k])

                if k==0:
                    ax1.plot(cal0.wvl_zen_si, cal0.primary_response_zen_si[intTimes_si[k]], label=label, color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i])
                    ax1.plot(cal0.wvl_zen_in, cal0.primary_response_zen_in[intTimes_in[k]], color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i])
                if k==1:
                    ax1.plot(cal0.wvl_zen_si, cal0.primary_response_zen_si[intTimes_si[k]], label=label, color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i], marker='o')
                    ax1.plot(cal0.wvl_zen_in, cal0.primary_response_zen_in[intTimes_in[k]], color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i], marker='o')


    ax1.legend(loc='upper right', fontsize=14, framealpha=0.4)

    ax1.set_title('Primary Response (Zenith 20180711)')
    ax1.set_xlim((250, 2250))
    ax1.set_ylim((0, 600))
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Primary Response')
    plt.savefig('pri_resp_20180711.png', bbox_inches='tight')
    plt.show()
    # ---------------------------------------------------------------------

def PLOT_TRANSFER_20180711():

    from ssfr_config import config_20180711_a1, config_20180711_a2, config_20180711_a3, config_20180711_b1, config_20180711_b2, config_20180711_b3

    markers     = ['D', '*']
    markersizes = [10, 4]
    linestyles = ['-', '--']
    colors     = ['red', 'blue', 'green']
    linewidths = [1.0, 1.0]
    alphas     = [1.0, 1.0]

    cal_a1 = CALIBRATION_CU_SSFR(config_20180711_a1)
    cal_a2 = CALIBRATION_CU_SSFR(config_20180711_a2)
    cal_a3 = CALIBRATION_CU_SSFR(config_20180711_a3)

    cals_a = [cal_a1, cal_a2, cal_a3]

    cal_b1 = CALIBRATION_CU_SSFR(config_20180711_b1)
    cal_b2 = CALIBRATION_CU_SSFR(config_20180711_b2)
    cal_b3 = CALIBRATION_CU_SSFR(config_20180711_b3)

    cals_b = [cal_b1, cal_b2, cal_b3]

    cals = [cals_a, cals_b]
    labels  = [['SSIM1 S45-90/I250-375', 'SSIM1 S45-45/I250-250', 'SSIM1 S90-90/I375-375'], ['10862 S45-90/I250-375', '10862 S45-45/I250-250', '10862 S90-90/I375-375']]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(111)
    for i, cals0 in enumerate(cals):
        for j, cal0 in enumerate(cals0):
            label0 = labels[i][j]

            # if 'SSIM1' in label0:
            intTimes_si = list(cal0.field_lamp_zen_si.keys())
            intTimes_in = list(cal0.field_lamp_zen_in.keys())

            for k in range(len(intTimes_si)):

                label = '%s (S%dI%d)' % (label0, intTimes_si[k], intTimes_in[k])

                if k==0:
                    ax1.plot(cal0.wvl_zen_si, cal0.field_lamp_zen_si[intTimes_si[k]], label=label, color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i])
                    ax1.plot(cal0.wvl_zen_in, cal0.field_lamp_zen_in[intTimes_in[k]], color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i])
                if k==1:
                    ax1.plot(cal0.wvl_zen_si, cal0.field_lamp_zen_si[intTimes_si[k]], label=label, color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i], marker='o')
                    ax1.plot(cal0.wvl_zen_in, cal0.field_lamp_zen_in[intTimes_in[k]], color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i], marker='o')


    ax1.legend(loc='upper right', fontsize=14, framealpha=0.4)

    ax1.set_title('Field Calibrator (Zenith 20180711)')
    ax1.set_xlim((250, 2250))
    ax1.set_ylim((0, 0.4))
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')
    plt.savefig('transfer_20180711.png', bbox_inches='tight')
    plt.show()
    # ---------------------------------------------------------------------

def PLOT_PRIMARY_RESPONSE_20180712():

    from ssfr_config import config_20180712_a1, config_20180712_a2, config_20180712_b1

    markers     = ['D', '*']
    markersizes = [10, 4]
    linestyles = ['-', '--']
    colors     = ['red', 'blue', 'green']
    linewidths = [1.0, 1.0]
    alphas     = [1.0, 1.0]

    cal_a1 = CALIBRATION_CU_SSFR(config_20180712_a1)
    cal_a2 = CALIBRATION_CU_SSFR(config_20180712_a2)
    cal_a3 = CALIBRATION_CU_SSFR(config_20180712_b1)

    cals = [[cal_a1, cal_a2, cal_a3]]
    labels  = [['2008-04 S45-90/I250-375', '2008-04 S90-150/I250-375', 'L2008-2 S45-90/I250-375']]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(111)
    for i, cals0 in enumerate(cals):
        for j, cal0 in enumerate(cals0):
            label0 = labels[i][j]

            intTimes_si = list(cal0.primary_response_nad_si.keys())
            intTimes_in = list(cal0.primary_response_nad_in.keys())

            for k in range(len(intTimes_si)):

                label = '%s (S%dI%d)' % (label0, intTimes_si[k], intTimes_in[k])

                if k==0:
                    ax1.plot(cal0.wvl_nad_si, cal0.primary_response_nad_si[intTimes_si[k]], label=label, color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i])
                    ax1.plot(cal0.wvl_nad_in, cal0.primary_response_nad_in[intTimes_in[k]], color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i])
                if k==1:
                    ax1.plot(cal0.wvl_nad_si, cal0.primary_response_nad_si[intTimes_si[k]], label=label, color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i], marker='o')
                    ax1.plot(cal0.wvl_nad_in, cal0.primary_response_nad_in[intTimes_in[k]], color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i], marker='o')


    ax1.legend(loc='upper right', fontsize=14, framealpha=0.4)

    ax1.set_title('Primary Response (Nadir 20180712)')
    ax1.set_xlim((250, 2250))
    ax1.set_ylim((0, 600))
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Primary Response')
    plt.savefig('pri_resp_20180712.png', bbox_inches='tight')
    plt.show()
    # ---------------------------------------------------------------------

def PLOT_TRANSFER_20180712():

    from ssfr_config import config_20180712_a1, config_20180712_a2, config_20180712_b1

    markers     = ['D', '*']
    markersizes = [10, 4]
    linestyles = ['-', '--']
    colors     = ['red', 'blue', 'green']
    linewidths = [1.0, 1.0]
    alphas     = [1.0, 1.0]

    cal_a1 = CALIBRATION_CU_SSFR(config_20180712_a1)
    cal_a2 = CALIBRATION_CU_SSFR(config_20180712_a2)
    cal_a3 = CALIBRATION_CU_SSFR(config_20180712_b1)

    cals = [[cal_a1, cal_a2, cal_a3]]
    labels  = [['2008-04 S45-90/I250-375', '2008-04 S90-150/I250-375', 'L2008-2 S45-90/I250-375']]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(111)
    for i, cals0 in enumerate(cals):
        for j, cal0 in enumerate(cals0):
            label0 = labels[i][j]

            intTimes_si = list(cal0.field_lamp_nad_si.keys())
            intTimes_in = list(cal0.field_lamp_nad_in.keys())

            for k in range(len(intTimes_si)):

                label = '%s (S%dI%d)' % (label0, intTimes_si[k], intTimes_in[k])

                if k==0:
                    ax1.plot(cal0.wvl_nad_si, cal0.field_lamp_nad_si[intTimes_si[k]], label=label, color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i])
                    ax1.plot(cal0.wvl_nad_in, cal0.field_lamp_nad_in[intTimes_in[k]], color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i])
                if k==1:
                    ax1.plot(cal0.wvl_nad_si, cal0.field_lamp_nad_si[intTimes_si[k]], label=label, color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i], marker='o')
                    ax1.plot(cal0.wvl_nad_in, cal0.field_lamp_nad_in[intTimes_in[k]], color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i], marker='o')


    ax1.legend(loc='upper right', fontsize=14, framealpha=0.4)

    ax1.set_title('Field Calibrator (Nadir 20180712)')
    ax1.set_xlim((250, 2250))
    ax1.set_ylim((0, 0.4))
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')
    plt.savefig('transfer_20180712.png', bbox_inches='tight')
    plt.show()
    # ---------------------------------------------------------------------

def PLOT_PRIMARY_RESPONSE_PRE_POST_NADIR():

    from ssfr_config_pre_post import config_20180320_n1, config_20180320_n2, config_20180712_n1, config_20180712_n2

    markers     = ['D', '*']
    markersizes = [10, 4]
    linestyles = ['-', '--']
    colors     = ['red', 'magenta', 'blue', 'cyan']
    linewidths = [1.0, 1.0]
    alphas     = [1.0, 1.0]

    cal_a1 = CALIBRATION_CU_SSFR(config_20180320_n1)
    cal_a2 = CALIBRATION_CU_SSFR(config_20180320_n2)
    cal_b1 = CALIBRATION_CU_SSFR(config_20180712_n1)
    cal_b2 = CALIBRATION_CU_SSFR(config_20180712_n2)

    cals = [[cal_a1, cal_a2, cal_b1, cal_b2]]
    labels  = [['Nadir 2008-04 (Pre. 20180320)', 'Nadir L2008-2 (Pre. 20180320)', 'Nadir 2008-04 (Post 20180712)', 'Nadir L2008-2 (Post 20180712)']]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(111)
    for i, cals0 in enumerate(cals):
        for j, cal0 in enumerate(cals0):
            label0 = labels[i][j]

            intTimes_si = list(cal0.primary_response_nad_si.keys())
            intTimes_in = list(cal0.primary_response_nad_in.keys())

            for k in range(len(intTimes_si)):

                label = '%s (S%dI%d)' % (label0, intTimes_si[k], intTimes_in[k])

                if k==0:
                    ax1.plot(cal0.wvl_nad_si, cal0.primary_response_nad_si[intTimes_si[k]], label=label, color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i])
                    ax1.plot(cal0.wvl_nad_in, cal0.primary_response_nad_in[intTimes_in[k]], color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i])
                if k==1:
                    ax1.plot(cal0.wvl_nad_si, cal0.primary_response_nad_si[intTimes_si[k]], label=label, color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i], marker='o')
                    ax1.plot(cal0.wvl_nad_in, cal0.primary_response_nad_in[intTimes_in[k]], color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i], marker='o')


    ax1.legend(loc='upper right', fontsize=14, framealpha=0.4)

    ax1.set_title('Primary Response (Nadir)')
    # ax1.set_xlim((250, 2250))
    # ax1.set_ylim((0, 600))
    ax1.set_xlim((350, 700))
    ax1.set_ylim((200, 500))
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Primary Response')
    # plt.savefig('pri_resp_pre_post_nadir.png', bbox_inches='tight')
    plt.savefig('pri_resp_pre_post_nadir_zoomed.png', bbox_inches='tight')
    plt.show()
    # ---------------------------------------------------------------------

def PLOT_PRIMARY_RESPONSE_PRE_POST_ZENITH():

    from ssfr_config_pre_post import config_20180320_z1, config_20180320_z2, config_20180711_z1, config_20180711_z2

    markers     = ['D', '*']
    markersizes = [10, 4]
    linestyles = ['-', '--']
    colors     = ['red', 'magenta', 'blue', 'cyan']
    linewidths = [1.0, 1.0]
    alphas     = [1.0, 1.0]

    cal_a1 = CALIBRATION_CU_SSFR(config_20180320_z1)
    cal_a2 = CALIBRATION_CU_SSFR(config_20180320_z2)
    cal_b1 = CALIBRATION_CU_SSFR(config_20180711_z1)
    cal_b2 = CALIBRATION_CU_SSFR(config_20180711_z2)

    cals = [[cal_a1, cal_a2, cal_b1, cal_b2]]
    labels  = [['Zenith SSIM1 (Pre. 20180320)', 'Zenith 108692-1 (Pre. 20180320)', 'Zenith SSIM1 (Post 20180711)', 'Zenith 108692-1 (Post 20180711)']]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(111)
    for i, cals0 in enumerate(cals):
        for j, cal0 in enumerate(cals0):
            label0 = labels[i][j]

            intTimes_si = list(cal0.primary_response_zen_si.keys())
            intTimes_in = list(cal0.primary_response_zen_in.keys())

            for k in range(len(intTimes_si)):

                label = '%s (S%dI%d)' % (label0, intTimes_si[k], intTimes_in[k])

                if k==0:
                    ax1.plot(cal0.wvl_zen_si, cal0.primary_response_zen_si[intTimes_si[k]], label=label, color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i])
                    ax1.plot(cal0.wvl_zen_in, cal0.primary_response_zen_in[intTimes_in[k]], color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i])
                if k==1:
                    ax1.plot(cal0.wvl_zen_si, cal0.primary_response_zen_si[intTimes_si[k]], label=label, color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i], marker='o')
                    ax1.plot(cal0.wvl_zen_in, cal0.primary_response_zen_in[intTimes_in[k]], color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i], marker='o')


    ax1.legend(loc='upper right', fontsize=14, framealpha=0.4)

    ax1.set_title('Primary Response (Zenith)')
    # ax1.set_xlim((250, 2250))
    # ax1.set_ylim((0, 600))
    ax1.set_xlim((350, 700))
    ax1.set_ylim((200, 500))
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Primary Response')
    # plt.savefig('pri_resp_pre_post_zenith.png', bbox_inches='tight')
    plt.savefig('pri_resp_pre_post_zenith_zoomed.png', bbox_inches='tight')
    plt.show()
    # ---------------------------------------------------------------------

def PLOT_TRANSFER_20180712():

    from ssfr_config import config_20180712_a1, config_20180712_a2, config_20180712_b1

    markers     = ['D', '*']
    markersizes = [10, 4]
    linestyles = ['-', '--']
    colors     = ['red', 'blue', 'green']
    linewidths = [1.0, 1.0]
    alphas     = [1.0, 1.0]

    cal_a1 = CALIBRATION_CU_SSFR(config_20180712_a1)
    cal_a2 = CALIBRATION_CU_SSFR(config_20180712_a2)
    cal_a3 = CALIBRATION_CU_SSFR(config_20180712_b1)

    cals = [[cal_a1, cal_a2, cal_a3]]
    labels  = [['2008-04 S45-90/I250-375', '2008-04 S90-150/I250-375', 'L2008-2 S45-90/I250-375']]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(111)
    for i, cals0 in enumerate(cals):
        for j, cal0 in enumerate(cals0):
            label0 = labels[i][j]

            intTimes_si = list(cal0.field_lamp_nad_si.keys())
            intTimes_in = list(cal0.field_lamp_nad_in.keys())

            for k in range(len(intTimes_si)):

                label = '%s (S%dI%d)' % (label0, intTimes_si[k], intTimes_in[k])

                if k==0:
                    ax1.plot(cal0.wvl_nad_si, cal0.field_lamp_nad_si[intTimes_si[k]], label=label, color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i])
                    ax1.plot(cal0.wvl_nad_in, cal0.field_lamp_nad_in[intTimes_in[k]], color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i])
                if k==1:
                    ax1.plot(cal0.wvl_nad_si, cal0.field_lamp_nad_si[intTimes_si[k]], label=label, color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i], marker='o')
                    ax1.plot(cal0.wvl_nad_in, cal0.field_lamp_nad_in[intTimes_in[k]], color=colors[j], lw=linewidths[i], alpha=alphas[i], ls=linestyles[i], marker='o')


    ax1.legend(loc='upper right', fontsize=14, framealpha=0.4)

    ax1.set_title('Field Calibrator (Nadir 20180712)')
    ax1.set_xlim((250, 2250))
    ax1.set_ylim((0, 0.4))
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Irradiance [$\mathrm{W m^{-2} nm^{-1}}$]')
    plt.savefig('transfer_20180712.png', bbox_inches='tight')
    plt.show()
    # ---------------------------------------------------------------------

def PLOT_PRIMARY_RESPONSE_20180719():

    from ssfr_config_skywatch_20180719 import config_20180719

    cal0 = CALIBRATION_CU_SSFR(config_20180719)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(16, 4))
    ax1 = fig.add_subplot(121)

    intTimes_si = list(cal0.primary_response_nad_si.keys())
    intTimes_in = list(cal0.primary_response_nad_in.keys())
    for k in range(len(intTimes_si)):
        label = 'S%dI%d' % (intTimes_si[k], intTimes_in[k])
        if k==0:
            ax1.plot(cal0.wvl_nad_si, cal0.primary_response_nad_si[intTimes_si[k]], label=label, color='r', lw=2.0)
            ax1.plot(cal0.wvl_nad_in, cal0.primary_response_nad_in[intTimes_in[k]], color='magenta', lw=2.0)
        if k==1:
            ax1.plot(cal0.wvl_nad_si, cal0.primary_response_nad_si[intTimes_si[k]], label=label, color='b', lw=2.0)
            ax1.plot(cal0.wvl_nad_in, cal0.primary_response_nad_in[intTimes_in[k]], color='cyan', lw=2.0)
    ax1.legend(loc='upper right', fontsize=14, framealpha=0.4)
    ax1.set_title('Primary Response (Nadir 2008-04)')
    ax1.set_xlim((250, 2250))
    ax1.set_ylim((0, 600))
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Response')

    ax2 = fig.add_subplot(122)
    intTimes_si = list(cal0.primary_response_zen_si.keys())
    intTimes_in = list(cal0.primary_response_zen_in.keys())
    for k in range(len(intTimes_si)):
        label = 'S%dI%d' % (intTimes_si[k], intTimes_in[k])
        if k==0:
            ax2.plot(cal0.wvl_zen_si, cal0.primary_response_zen_si[intTimes_si[k]], label=label, color='r', lw=2.0)
            ax2.plot(cal0.wvl_zen_in, cal0.primary_response_zen_in[intTimes_in[k]], color='magenta', lw=2.0)
        if k==1:
            ax2.plot(cal0.wvl_zen_si, cal0.primary_response_zen_si[intTimes_si[k]], label=label, color='b', lw=2.0)
            ax2.plot(cal0.wvl_zen_in, cal0.primary_response_zen_in[intTimes_in[k]], color='cyan', lw=2.0)
    ax2.legend(loc='upper right', fontsize=14, framealpha=0.4)
    ax2.set_title('Primary Response (Zenith SSIM1)')
    ax2.set_xlim((250, 2250))
    ax2.set_ylim((0, 600))
    ax2.set_xlabel('Wavelength [nm]')
    ax2.set_ylabel('Response')


    plt.savefig('pri_resp_20180719.png', bbox_inches='tight')
    plt.show()
    # ---------------------------------------------------------------------

def PLOT_TRANSFER_20180719():

    from ssfr_config_skywatch_20180719 import config_20180719

    cal0 = CALIBRATION_CU_SSFR(config_20180719)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(16, 4))
    ax1 = fig.add_subplot(121)

    intTimes_si = list(cal0.field_lamp_nad_si.keys())
    intTimes_in = list(cal0.field_lamp_nad_in.keys())
    for k in range(len(intTimes_si)):
        label = 'S%dI%d' % (intTimes_si[k], intTimes_in[k])
        if k==0:
            ax1.plot(cal0.wvl_nad_si, cal0.field_lamp_nad_si[intTimes_si[k]], label=label, color='r', lw=2.0)
            ax1.plot(cal0.wvl_nad_in, cal0.field_lamp_nad_in[intTimes_in[k]], color='magenta', lw=2.0)
        if k==1:
            ax1.plot(cal0.wvl_nad_si, cal0.field_lamp_nad_si[intTimes_si[k]], label=label, color='b', lw=2.0)
            ax1.plot(cal0.wvl_nad_in, cal0.field_lamp_nad_in[intTimes_in[k]], color='cyan', lw=2.0)
    ax1.legend(loc='upper right', fontsize=14, framealpha=0.4)
    ax1.set_title('Field Lamp (Nadir 2008-04)')
    ax1.set_xlim((250, 2250))
    ax1.set_ylim((0, 0.4))
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Flux [$\mathrm{W m^{-2} nm^{-1}}$]')

    ax2 = fig.add_subplot(122)
    intTimes_si = list(cal0.field_lamp_zen_si.keys())
    intTimes_in = list(cal0.field_lamp_zen_in.keys())
    for k in range(len(intTimes_si)):
        label = 'S%dI%d' % (intTimes_si[k], intTimes_in[k])
        if k==0:
            ax2.plot(cal0.wvl_zen_si, cal0.field_lamp_zen_si[intTimes_si[k]], label=label, color='r', lw=2.0)
            ax2.plot(cal0.wvl_zen_in, cal0.field_lamp_zen_in[intTimes_in[k]], color='magenta', lw=2.0)
        if k==1:
            ax2.plot(cal0.wvl_zen_si, cal0.field_lamp_zen_si[intTimes_si[k]], label=label, color='b', lw=2.0)
            ax2.plot(cal0.wvl_zen_in, cal0.field_lamp_zen_in[intTimes_in[k]], color='cyan', lw=2.0)
    ax2.legend(loc='upper right', fontsize=14, framealpha=0.4)
    ax2.set_title('Field Lamp (Zenith SSIM1)')
    ax2.set_xlim((250, 2250))
    ax2.set_ylim((0, 0.4))
    ax2.set_xlabel('Wavelength [nm]')
    ax2.set_ylabel('Flux [$\mathrm{W m^{-2} nm^{-1}}$]')

    plt.savefig('transfer_20180719.png', bbox_inches='tight')
    plt.show()
    # ---------------------------------------------------------------------

def PLOT_SECONDARY_RESPONSE_20180719():

    from ssfr_config_skywatch_20180719 import config_20180719

    cal0 = CALIBRATION_CU_SSFR(config_20180719)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(16, 4))
    ax1 = fig.add_subplot(121)

    intTimes_si = list(cal0.secondary_response_nad_si.keys())
    intTimes_in = list(cal0.secondary_response_nad_in.keys())
    for k in range(len(intTimes_si)):
        label = 'S%dI%d' % (intTimes_si[k], intTimes_in[k])
        if k==0:
            ax1.plot(cal0.wvl_nad_si, cal0.secondary_response_nad_si[intTimes_si[k]], label=label, color='r', lw=2.0)
            ax1.plot(cal0.wvl_nad_in, cal0.secondary_response_nad_in[intTimes_in[k]], color='magenta', lw=2.0)
        if k==1:
            ax1.plot(cal0.wvl_nad_si, cal0.secondary_response_nad_si[intTimes_si[k]], label=label, color='b', lw=2.0)
            ax1.plot(cal0.wvl_nad_in, cal0.secondary_response_nad_in[intTimes_in[k]], color='cyan', lw=2.0)
    ax1.legend(loc='upper right', fontsize=14, framealpha=0.4)
    ax1.set_title('Secondary Response (Nadir 2008-04)')
    ax1.set_xlim((250, 2250))
    ax1.set_ylim((0, 600))
    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Response')

    ax2 = fig.add_subplot(122)
    intTimes_si = list(cal0.secondary_response_zen_si.keys())
    intTimes_in = list(cal0.secondary_response_zen_in.keys())
    for k in range(len(intTimes_si)):
        label = 'S%dI%d' % (intTimes_si[k], intTimes_in[k])
        if k==0:
            ax2.plot(cal0.wvl_zen_si, cal0.secondary_response_zen_si[intTimes_si[k]], label=label, color='r', lw=2.0)
            ax2.plot(cal0.wvl_zen_in, cal0.secondary_response_zen_in[intTimes_in[k]], color='magenta', lw=2.0)
        if k==1:
            ax2.plot(cal0.wvl_zen_si, cal0.secondary_response_zen_si[intTimes_si[k]], label=label, color='b', lw=2.0)
            ax2.plot(cal0.wvl_zen_in, cal0.secondary_response_zen_in[intTimes_in[k]], color='cyan', lw=2.0)
    ax2.legend(loc='upper right', fontsize=14, framealpha=0.4)
    ax2.set_title('Secondary Response (Zenith SSIM1)')
    ax2.set_xlim((250, 2250))
    ax2.set_ylim((0, 600))
    ax2.set_xlabel('Wavelength [nm]')
    ax2.set_ylabel('Response')

    plt.savefig('sec_resp_20180719.png', bbox_inches='tight')
    plt.show()
    # ---------------------------------------------------------------------

def CDATA_20180719():

    from ssfr_config_skywatch_20180719 import config_20180719

    cal = CALIBRATION_CU_SSFR(config_20180719)

    date = datetime.datetime(2018, 7, 19)
    # read in data
    # ==============================================================
    fdir   = '/Users/hoch4240/Chen/work/07_ORACLES-2/filtered-spn/data/SSFR/20180719/Alvin/s45_90i250_375'
    # ==============================================================
    fnames = sorted(glob.glob('%s/*.SKS' % fdir))
    ssfr   = CU_SSFR(fnames)
    # ==============================================================
    whichRadiation = {'zenith':'irradiance', 'nadir':'irradiance'}
    # ==============================================================
    ssfr.COUNT2RADIATION(cal, whichRadiation=whichRadiation)

    f = h5py.File('%s_Alvin.h5' % date.strftime('%Y%m%d'), 'w')
    f['spectra_zen'] = ssfr.spectra_zen
    f['spectra_nad'] = ssfr.spectra_nad
    f['wvl_zen'] = ssfr.wvl_zen
    f['wvl_nad'] = ssfr.wvl_nad
    f['tmhr']    = ssfr.tmhr
    f['temp']    = ssfr.temp
    f.close()
    exit()


