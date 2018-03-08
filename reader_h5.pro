pro read_cu_ssfr_h5

  fname = '20180228_CU_20180221.h5'

  f_id = H5F_OPEN(fname)

  d_id = H5D_OPEN(f_id, 'tmhr')
  tmhr = H5D_READ(d_id)
  H5D_CLOSE, d_id

  d_id = H5D_OPEN(f_id, 'shutter')
  shutter = H5D_READ(d_id)
  H5D_CLOSE, d_id

  d_id = H5D_OPEN(f_id, 'wvl_nad')
  wvl_nad = H5D_READ(d_id)
  H5D_CLOSE, d_id

  d_id = H5D_OPEN(f_id, 'wvl_zen')
  wvl_zen = H5D_READ(d_id)
  H5D_CLOSE, d_id

  d_id = H5D_OPEN(f_id, 'spectra_flux_nad')
  spectra_flux_nad = H5D_READ(d_id)
  H5D_CLOSE, d_id

  d_id = H5D_OPEN(f_id, 'spectra_flux_zen')
  spectra_flux_zen = H5D_READ(d_id)
  H5D_CLOSE, d_id

  H5F_CLOSE, f_id

  ; now the avaiable variables are:
  ; tmhr
  ; shutter
  ; wvl_nad
  ; wvl_zen
  ; spectra_flux_nad
  ; spectra_flux_zen

end
