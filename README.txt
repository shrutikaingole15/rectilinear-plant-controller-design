AE 308/775 Course Project - Group 22 (P2, ECP rectilinear plant)
=================================================================

Requirements : Python 3.10+, numpy, scipy, matplotlib
Run          : python3 main.py        (from inside this folder)
Output       : fig/*.png  (all report figures, titles carry "Group 22")
               results.json  (every number quoted in the report)

Files
  main.py   top-level script, regenerates everything
  ctl.py    helper functions (TF algebra, margins, ||S||inf, bandwidth,
            step-response metrics, control effort)

Figure -> script section (all in main.py)
  pzmap.png, ol_step.png, ol_bode.png            "Part A figures"
  rlocus_P.png, nyquist_P.png                    "Routh / RL / Nyquist"
  region.png                                     "Part B(1)-(2)"
  P_step.png                                     "Part B(3) baseline P"
  PI_bode.png, PI_rlocus.png                     "Design 1: PI" + "design figures"
  LAG_bode.png, LAG_rlocus.png                   "Design 2: lag compensator" + "design figures"
  compare_step_u.png, u_zoom.png, sens.png       "step + effort comparison", "sensitivity"
  PI_nyquist.png, clpoles.png, PI_dominant.png   validation section
  setup.jpg, schematic.png                       taken from the interim report (not generated)
