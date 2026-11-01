# References

Every equation and material value in the code cites one of these. Numbers in brackets are used in docstrings and figure captions.

## Method: dispersion relation, Riemann sheets, root finding
- **[Chew]** W. C. Chew, *Waves and Fields in Inhomogeneous Media* (Van Nostrand Reinhold 1990; IEEE Press 1995), ch. 2 — transfer matrices and Sommerfeld integrals for layered media.
- **[SHF92]** R. E. Smith, S. N. Houde-Walter, G. W. Forbes, "Mode determination for planar waveguide using the four-sheeted dispersion relation," *IEEE J. Quantum Electron.* **28**(6), 1520–1526 (1992). doi:10.1109/3.135305 — four-sheeted Riemann surface of the planar-guide dispersion function; finding proper and improper modes of lossy multilayers.
- **[AG92]** E. Anemogiannis, E. N. Glytsis, "Multilayer waveguides: efficient numerical analysis of general structures," *J. Lightwave Technol.* **10**(10), 1344–1351 (1992) — argument-principle method (APM) for multilayer guides.
- **[DL67]** L. M. Delves, J. N. Lyness, "A numerical method for locating the zeros of an analytic function," *Math. Comput.* **21**(100), 543–560 (1967) — contour counting and moments.
- **[KVB99]** P. Kravanja, M. Van Barel, "A derivative-free algorithm for computing zeros of analytic functions," *Computing* **63**(1), 69–91 (1999).
- **[KVMF09]** Ş. E. Kocabaş, G. Veronis, D. A. B. Miller, S. Fan, "Modal analysis and coupling in metal-insulator-metal waveguides," *Phys. Rev. B* **79**, 035120 (2009). arXiv:0809.2850 — complete (bound + leaky + continuum) modal picture of MIM guides. Kocabaş's argument-principle Mathematica notebook for this guide, https://kocabas.gitlab.io/publications/code/ModalShapes_Plasmon.nb, is the basis of the author's notebook, which `apm.py` ports.
- **[HM09]** J. Hu, C. R. Menyuk, "Understanding leaky modes: slab waveguide revisited," *Adv. Opt. Photon.* **1**, 58–106 (2009) — proper/improper modes and their physical meaning.

## Dipoles over layered media
- **[MM97]** K. A. Michalski, J. R. Mosig, "Multilayered media Green's functions in integral equation formulations," *IEEE Trans. Antennas Propag.* **45**(3), 508–519 (1997).
- **[Balanis]** C. A. Balanis, *Antenna Theory: Analysis and Design*, 4th ed. (Wiley 2016), §4.7 (vertical dipole above a ground plane) and §2.17 (Friis).
- **[Friis46]** H. T. Friis, "A note on a simple transmission formula," *Proc. IRE* **34**, 254–256 (1946).

## On-body / intra-body propagation context
- **[Asan18]** N. B. Asan et al., "Characterization of the fat channel for intra-body communication at R-band frequencies," *Sensors* **18**(9), 2752 (2018).
- **[Asan17]** N. B. Asan et al., "Intra-body microwave communication through adipose tissue," *Healthc. Technol. Lett.* **4**(4), 115–121 (2017). doi:10.1049/htl.2016.0104
- **[GM14]** M. Grimm, D. Manteuffel, "Norton surface waves in the scope of body area networks," *IEEE Trans. Antennas Propag.* (2014) — Norton surface wave as the on-body mechanism for flush-mounted antennas.
- **[CJ13]** R. Chandra, A. J. Johansson, "A link loss model for the on-body propagation channel for binaural hearing aids," *IEEE Trans. Antennas Propag.* **61**(12), 6180–6190 (2013). doi:10.1109/TAP.2013.2283270, arXiv:1304.1739
- **[Kvist]** S. H. Kvist, J. Thaysen, K. B. Jakobsen, ear-to-ear on-body channel papers (DTU), see `materials.py` for the specific values used.
- **[Tian19]** X. Tian et al., "Wireless body sensor networks based on metamaterial textiles," *Nat. Electron.* **2**, 243–251 (2019).
- **[Col04]** R. E. Collin, "Hertzian dipole radiating over a lossy earth or sea: some early and late 20th-century controversies," *IEEE Antennas Propag. Mag.* **46**(2), 64–79 (2004). doi:10.1109/MAP.2004.1305535 — Zenneck pole vs Norton wave (the cancellation seen in our flat-body decomposition).
- **[HH]** P. S. Hall, Y. Hao, body-centric wireless communications article, *Microwave Journal* (title as on the linked PDF) — creeping-wave attenuation ≈ 2 dB/cm at 2.4 GHz (citing Ryckaert et al.); torso shadow-region fit PG = −0.36 d[cm] − 35 dB. https://www.microwavejournal.com/ext/resources/BGDownload/9/2/bodycentric.pdf
- **[Wang18]** arXiv:1801.09224 (2018) — through-body loss ≈ 120 dB over a 30 cm path at 2.4 GHz (secondary citation).

## Ear-to-ear measurements and models (2.45 GHz)
- **[RTJ16]** A. Ruaro, J. Thaysen, K. B. Jakobsen, *IEEE Trans. Antennas Propag.* (2016) — in-the-ear antennas on SAM: simulated max −54.4 dB, measured average −66.1 dB. https://backend.orbit.dtu.dk/ws/files/123683538/07436778.pdf
- **[Kvist10]** S. H. Kvist et al., LAPC 2010 — SAM phantom, normal monopoles outside the ear: −35 to −48 dB measured. https://orbit.dtu.dk/files/5588418/74C11B08d01.pdf
- **[CJ13]** (above) — link loss 70.25 dB (Billie) and 81.7 dB (Duke) incl. pinna; pinnae add 7–11 dB.
- **[Kam18]** S. Kammersgaard, *Wireless communication for custom hearing instruments*, PhD thesis, DTU (2018) — "better than −80 dB" for realistic in-the-ear devices.
- **[NKJ23]** Nielsen, Kvist, Jakobsen, EuCAP 2023 — ear-to-ear geodesics: back 2.12 λ, front 2.77 λ, over the top 3.46 λ; optimised path gain −35 dB with lossless antennas. https://backend.orbit.dtu.dk/ws/portalfiles/portal/325050721/Ear_to_Ear_Propagation_ModelAntenna_Radiation_Pattern_Optimization.pdf

## Material and geometry data
- **[G96]** S. Gabriel, R. W. Lau, C. Gabriel, "The dielectric properties of biological tissues: III. Parametric models for the dielectric spectrum of tissues," *Phys. Med. Biol.* **41**, 2271–2293 (1996) — 4-pole Cole-Cole parameters (Table 1).
- **[ITIS]** P. A. Hasgall et al., IT'IS Database for thermal and electromagnetic parameters of biological tissues, v4.1 (2022) — uses the [G96] model. https://itis.swiss/assets/Downloads/TissueDb/V4-1/Documentation-v4-1.pdf
- **[IFAC]** IFAC-CNR, Gabriel 4-pole parameters, Appendix C: https://niremf.ifac.cnr.it/docs/DIELECTRIC/AppendixC.html
- **[OSSDBS]** OSS-DBS v2, `ossdbs/dielectric_model/colecole4.py` (CSF Cole-Cole parameters). https://github.com/SFB-ELAINE/OSS-DBSv2
- **[RDH19]** E. A. Rashed, Y. Diao, A. Hirata, arXiv:1911.01220 (2019), Table 1 — Gabriel-based values used to verify dura.
- **[NICT]** NICT dielectric property database, "Data of Dielectric Measurements – Tendon" (animal by-products, 30–37 °C): ε′ ≈ 51, σ ≈ 1.9 S/m at 2.4–2.5 GHz, compared with 43 / 1.69 S/m from the [G96] tendon model used here. https://www2.nict.go.jp/cgi-bin/202303080003/public_html/index.py/measurements/Tendon
- **[Pirri21]** C. Pirri et al., *Front. Rehabil. Sci.* **2**, 743553 (2021) — temporalis 8.1 ± 1.6 mm, deep temporal fascia 0.60 ± 0.20 mm (ultrasound).
- **[Pir22]** C. Pirri et al., "Ultrasound Imaging of the Superficial Fascia in the Upper Limb: Arm and Forearm," *Diagnostics* **12**(8), 1884 (2022). Forearm superficial fascia 0.36–0.44 mm; reports no subcutaneous fat thickness, so the wrist fat values here are assumed.
- **[Bhushal25]** Bhushal et al., *J. Inst. Med. Nepal* (2025) — temporal squama 1.52 ± 0.50 mm (cadaver, n = 72). https://www.nepjol.info/index.php/JIOM/article/view/92990
- **[DelBrutto16]** O. H. Del Brutto et al., *J. Clin. Ultrasound* (2016) — temporal bone at the TCD window 2.7 ± 0.9 mm (good) / 4.7 ± 1.2 mm (poor).
- **[Saban13]** Sabanciogullari et al., *Int. J. Morphol.* **31**, 38 (2013) — skull tables/diploë at opisthocranion, lambda, bregma, euryon (MRI, n = 220).
- **[Stö18]** Störchle et al., "Measurement of mean subcutaneous fat thickness: eight standardised ultrasound sites compared to 216 randomly selected sites," *Sci. Rep.* **8**, 16268 (2018).
- **[Law14]** Lawrence, Branson, Oliva, Rubinowitz, "MDCT of the central airways: anatomy and pathology," *Applied Radiology* **43**(11), 8–21 (2014).
- **[NIOSH04]** NIOSH / Anthrotech head-and-face anthropometric survey (2004), Table 8 — head breadth, circumference, bitragion arcs, stature. https://nap.nationalacademies.org/resource/11815/Anthrotech_report.pdf
- **[ANSUR2]** ANSUR II summary tables (NCSU ergonomics center) — standing eye height. https://www.ergocenter.ncsu.edu/wp-content/uploads/sites/18/2017/09/Anthropometric-Summary-Data-Tables.pdf
- **[P2040]** ITU-R P.2040-3, Table 3 (ε_r = a f^b, σ = c f^d), as reproduced by MathWorks: https://www.mathworks.com/help/comm/ref/buildingmaterialpermittivity.html
- **[nRF5340]** Nordic Semiconductor nRF5340 product brief — TX +3 to −20 dBm; sensitivity −98 dBm (LE 1M), −95 dBm (LE 2M).

**Not found on a fetched page (marked `ASSUMED` in `materials.py`):** temple skin and subcutaneous-fat thickness, scalp skin/fat thickness, CSF and dura thickness, bitragion breadth, seated ear height.

## Loss / gain guiding
- **[Sie03]** A. E. Siegman, "Propagating modes in gain-guided optical fibers," *JOSA A* **20**(8), 1617–1628 (2003).
- **[Sie07]** A. E. Siegman, "Gain-guided, index-antiguided fiber lasers," *JOSA B* **24**(8), 1677–1682 (2007).
- **[Sie06]** A. E. Siegman, Y. Chen, V. Sudesh, M. C. Richardson, M. Bass, P. Foy, W. Hawkins, J. Ballato, "Confined propagation and near single-mode laser oscillation in a gain-guided, index antiguided optical fiber," *Appl. Phys. Lett.* **89**, 251101 (2006).
- **[Che07]** Y. Chen, V. Sudesh, T. McComb, M. C. Richardson, M. Bass, J. Ballato, "Lasing in a gain-guided index antiguided fiber," *JOSA B* **24**(8), 1683–1688 (2007).
- **[Kog65]** H. Kogelnik, "On the propagation of Gaussian beams of light through lenslike media including those with a loss or gain variation," *Appl. Opt.* **4**, 1562–1569 (1965). (not opened)
- **[CN75]** D. D. Cook, F. R. Nash, "Gain-induced guiding and astigmatic output beam of GaAs lasers," *J. Appl. Phys.* **46**, 1660 (1975). (not opened)
- **[Pet79]** K. Petermann, "Calculated spontaneous emission factor for double-heterostructure injection lasers with gain-induced waveguiding," *IEEE J. Quantum Electron.* **15**(7), 566–570 (1979).
- **[MS64]** E. A. J. Marcatili, R. A. Schmeltzer, "Hollow metallic and dielectric waveguides for long distance optical transmission and lasers," *Bell Syst. Tech. J.* **43**, 1783–1809 (1964).
- **[Dug86]** M. A. Duguay, Y. Kokubun, T. L. Koch, L. Pfeiffer, "Antiresonant reflecting optical waveguides in SiO₂–Si multilayer structures," *Appl. Phys. Lett.* **49**, 13–15 (1986).
- **[SC79]** H. K. Sacks, R. L. Chufo, *Medium-Frequency Propagation in Coal Mines*, US Bureau of Mines (c. 1979); with A. G. Emslie, R. L. Lagace (1978 Boulder workshop) and J. R. Wait, USBM report H0155008 (1976). https://cdc.gov/NIOSH/mining/UserFiles/works/pdfs/mfpicm.pdf
- **[SCG16]** S. Savoia, G. Castaldi, V. Galdi, "Non-Hermiticity-induced wave confinement and guiding in loss-gain-loss three-layer systems," *Phys. Rev. A* **94**, 043838 (2016). arXiv:1610.01135
