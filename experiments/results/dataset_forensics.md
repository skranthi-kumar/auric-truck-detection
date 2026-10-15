# Dataset forensics

## A. Label validity
- train: 7618 boxes; class ids [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)]; invalid flag 0; side < 4 px: 0; side > 150 px: 0; aspect > 4: 288; outside image: 0
- val: 1552 boxes; class ids [np.int64(0), np.int64(1), np.int64(2), np.int64(3), np.int64(4)]; invalid flag 0; side < 4 px: 2; side > 150 px: 1; aspect > 4: 41; outside image: 0
- empty images: train 1, val 0

## B. Duplicate and conflicting boxes (per image)
- train: same-class IoU>=0.7 (duplicate labels) 14 (0.2%); different-class IoU>=0.5 (one object, two labels) 8 (0.1%); same-class IoU>=0.5 (class-wise NMS 0.5 can keep only one) 53 (0.7%)
- val: same-class IoU>=0.7 (duplicate labels) 0 (0.0%); different-class IoU>=0.5 (one object, two labels) 0 (0.0%); same-class IoU>=0.5 (class-wise NMS 0.5 can keep only one) 27 (1.7%)

## C. Size and aspect by class (px, native resolution)
                count   10%   50%   90%
split cls                              
train Cargo    3773.0  12.0  18.5  30.3
      Box      2366.0  19.1  31.5  45.9
      Flatbed   662.0  14.8  25.9  41.6
      Tractor   625.0  12.4  18.7  31.9
      Liquid    192.0  15.3  25.5  38.7
val   Cargo     800.0  12.0  22.4  35.6
      Box       493.0  17.1  33.5  48.8
      Flatbed   122.0  18.0  28.5  39.1
      Tractor   117.0  11.7  21.0  33.5
      Liquid     20.0  20.4  41.5  58.5

- median aspect (train): Cargo 1.62, Box 2.00, Flatbed 1.75, Tractor 1.62, Liquid 1.65
- size-only classifier (nearest class median side) val accuracy: 0.407, majority-class baseline: 0.515

## D. Train vs val scale shift (per-image median box side)
- train: per-image median side  10% 14.5  50% 21.8  90% 35.3 px
- val: per-image median side  10% 16.5  50% 24.5  90% 33.9 px
- largest val images: 2308 (7204x5458, 76 boxes, median side 31.7), 2391 (7072x5932, 108 boxes, median side 30.9), 1929 (4979x3183, 77 boxes, median side 21.2)

## E. Class concentration (how few images hold a class)

train:
  class  boxes  images top-3 images hold                       top-3
  Cargo   3773     368               11%   1127:248, 109:84, 2370:79
    Box   2366     261               28% 2459:338, 112:205, 1058:118
Flatbed    662     195               18%   1127:47, 2322:39, 1896:35
Tractor    625     128               44%  1896:168, 1127:67, 2561:37
 Liquid    192      99               23%       1127:21, 2303:14, 5:9

val:
  class  boxes  images top-3 images hold                       top-3
  Cargo    800      19               45% 1211:153, 2293:113, 1206:91
    Box    493      17               58%  2470:149, 2472:75, 2391:63
Flatbed    122      14               66%   1447:35, 2293:28, 2292:18
Tractor    117      11               59%   2472:26, 2543:22, 1181:21
 Liquid     20       8               60%      1362:6, 2308:4, 1399:2

## F. Per-image class purity (share of an image's boxes in its majority class)
- train (images with >=5 boxes, n=263): purity median 0.70, >=0.9 in 19% of images
- val (images with >=5 boxes, n=22): purity median 0.77, >=0.9 in 32% of images

## G. Context oracle: can neighbours predict a box's class? (leave-one-out, val GT)
- neighbour-majority (radius 150 px): accuracy 0.890 on 1373 boxes with a neighbour
- neighbour-majority (radius 400 px): accuracy 0.810 on 1507 boxes with a neighbour
- neighbour-majority (whole image): accuracy 0.774 on 1552 boxes with a neighbour
  (reference: crop classifiers on GT boxes 0.57-0.59; majority class 0.52)

## H. Border and crowding
- train: touching border 1.2%; overlapping another GT (IoU>0) 37.0%; IoU>0.3 6.7%
- val: touching border 1.1%; overlapping another GT (IoU>0) 48.6%; IoU>0.3 10.1%

## I. Val under-annotation proxy (E1_up2_s0: confident background FPs vs GT per image)
        gt  bgFP>=0.5  ratio
image                       
31      35          8   0.23
1206    97         17   0.18
1457    13          1   0.08
1530    14          1   0.07
1362    16          1   0.06
1211   167          7   0.04
2308    76          2   0.03
2292    30          1   0.03

- total confident bg FPs 49 vs 1552 GT; images with ratio >= 0.5: 0

## J. Instances per image
- train: boxes/image  min 0  median 7  90% 41  max 451
- val: boxes/image  min 8  median 76  90% 141  max 167
