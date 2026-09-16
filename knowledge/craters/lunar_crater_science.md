# Lunar Crater Science Reference

## Crater Morphology
Simple craters (diameter < 15km): bowl-shaped, depth/diameter ~0.2.
Complex craters (> 15km): central peak, terraced walls, flat floor.
Degraded craters: reduced rim height, infilled floor — indicates age.

## Size-Frequency Distribution (SFD)
Power-law: N(>D) = k * D^(-b). Slope b ~ 2-3 for primary craters.
Steeper slope (b > 3) indicates secondary cratering dominance.
Crater density per km2 can estimate relative surface age using
published Neukum/Ivanov chronology functions.

## Registration Significance
Cross-sensor crater matching validates registration quality independently
of SIFT tie-points. If a crater detected in OHRC image spatially matches
the same crater in LROC NAC after registration, this is strong independent
confirmation that the geometric transform is correct.
Mismatch > 2px between corresponding crater centers = registration suspect.
