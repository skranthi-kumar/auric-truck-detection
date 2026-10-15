## Runs
             run kind  frac  images  instances  mAP50  Cargo   Box  Flatbed  Tractor  Liquid
LC_full_1000_s0 full 1.000     442       7618  0.088  0.159 0.170    0.090    0.013   0.008
LC_rand_0125_s0 rand 0.125      59        965  0.052  0.113 0.142    0.006    0.000   0.000
LC_rand_0125_s1 rand 0.125      59        860  0.028  0.089 0.026    0.025    0.000   0.000
LC_rand_0250_s0 rand 0.250     114       1541  0.051  0.113 0.115    0.024    0.001   0.000
LC_rand_0250_s1 rand 0.250     114       1637  0.054  0.123 0.069    0.043    0.002   0.031
LC_rand_0500_s0 rand 0.500     223       3682  0.064  0.145 0.112    0.056    0.007   0.001
 LC_sel_0125_s0  sel 0.125      67        841  0.047  0.083 0.104    0.048    0.002   0.000
 LC_sel_0125_s1  sel 0.125      67        841  0.032  0.077 0.069    0.014    0.002   0.000
 LC_sel_0250_s0  sel 0.250     120       1380  0.033  0.066 0.062    0.029    0.010   0.000
 LC_sel_0250_s1  sel 0.250     120       1380  0.041  0.073 0.094    0.034    0.003   0.000

## 5.3 learning curves (random subsets; seed means)
       instances  mAP50  Cargo    Box  Flatbed  Tractor  Liquid
frac                                                           
0.125      912.5  0.040  0.101  0.084    0.015    0.000   0.000
0.250     1589.0  0.052  0.118  0.092    0.033    0.001   0.016
0.500     3682.0  0.064  0.145  0.112    0.056    0.007   0.001
1.000     7618.0  0.088  0.159  0.170    0.090    0.013   0.008
seed spread of mAP50 at each fraction: {0.125: 0.024, 0.25: 0.003, 0.5: 0.0}

+500 instances = +6.6% of 7618; per-class share of the +500 (current mix): Cargo 247, Box 155, Flatbed 43, Tractor 41, Liquid 12
  mAP50    fit a=1.000 b=1.13 c=0.02  AP(full)=0.085 -> +500: +0.0013  (max |resid| 0.005)
  Cargo    fit a=0.235 b=0.88 c=0.28  AP(full)=0.160 -> +500: +0.0013  (max |resid| 0.002)
  Box      fit a=1.000 b=1.25 c=0.04  AP(full)=0.157 -> +500: +0.0023  (max |resid| 0.017)
  Flatbed  fit a=1.000 b=1.26 c=0.04  AP(full)=0.086 -> +500: +0.0021  (max |resid| 0.006)
  Tractor  fit a=0.635 b=0.68 c=0.01  AP(full)=0.012 -> +500: +0.0004  (max |resid| 0.001)
  Liquid   fit a=0.008 b=50.00 c=1.38  AP(full)=0.008 -> +500: +0.0000  (max |resid| 0.010)
  Caveats: 4 points, 1-2 seeds; extrapolation is a few % beyond the largest point; a flat low curve points to non-sample limits (see 5.1 ceiling, 5.2 hard-confused).

## 5.4 selected vs random (same instance budget)
  0.125: random 0.040 (range 0.028-0.052, 912 inst, 59 img) | selected 0.040 (range 0.032-0.047, 841 inst, 67 img) | gap -0.000; exceeds both seed spreads: False | >= 90% of full (0.079): random False, selected False
  0.250: random 0.052 (range 0.051-0.054, 1589 inst, 114 img) | selected 0.037 (range 0.033-0.041, 1380 inst, 120 img) | gap -0.015; exceeds both seed spreads: True | >= 90% of full (0.079): random False, selected False
  smallest tested subset reaching 90% of full-data mAP50: none of the tested sizes

## 5.4 what survives selection (12.5%)
  selected  : 67 images, boxes/image 12.6, region shares easy 0.13 ambiguous 0.48 hard-confused 0.04 hard-missed 0.01, median side 28.8 px
  random s0 : 59 images, boxes/image 16.4, region shares easy 0.11 ambiguous 0.19 hard-confused 0.16 hard-missed 0.11, median side 23.2 px
  random s1 : 59 images, boxes/image 14.6, region shares easy 0.04 ambiguous 0.18 hard-confused 0.17 hard-missed 0.10, median side 19.8 px
