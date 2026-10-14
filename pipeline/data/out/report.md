# Price model report

> **DEV MODE:** prices are synthetic (free sample has prices redacted). These numbers only prove the pipeline works; rerun on paid data for real accuracy.

## Test-set accuracy

| metric | our model | CarGurus expectedPrice |
|---|---|---|
| rows | 4491 | 4491 |
| MAE ($) | 1975.0 | 1284.0 |
| MAPE (%) | 12.7 | 9.2 |
| median error (%) | 8.7 | 5.3 |
| within 10% (%) | 55.3 | 72.2 |
| within 20% (%) | 82.6 | 89.1 |

Target median error <= 10%: **MET**

Note: the target is the *asking* price. Some error is unavoidable because the same car is listed as a great deal by one dealer and overpriced by another.

## Error by make

| make          |   rows |   median error (%) |
|:--------------|-------:|-------------------:|
| Porsche       |     24 |               16.4 |
| Dodge         |    100 |               11.4 |
| Volvo         |     37 |               11   |
| BMW           |    141 |               10   |
| Lincoln       |     42 |                9.9 |
| Audi          |     95 |                9.7 |
| GMC           |    168 |                9.7 |
| Toyota        |    457 |                9.6 |
| Nissan        |    280 |                9.6 |
| INFINITI      |     27 |                9.4 |
| Mercedes-Benz |    120 |                9   |
| Lexus         |    104 |                9   |
| Buick         |     49 |                9   |
| Mazda         |    121 |                8.9 |
| Subaru        |    145 |                8.8 |
| Chevrolet     |    495 |                8.7 |
| Cadillac      |     69 |                8.7 |
| Ford          |    540 |                8.6 |
| Mitsubishi    |     21 |                8.5 |
| Acura         |     50 |                8.4 |
| Chrysler      |     48 |                8.2 |
| Kia           |    130 |                8.2 |
| RAM           |    155 |                8   |
| Honda         |    327 |                7.8 |
| Volkswagen    |    122 |                7.7 |
| Jeep          |    260 |                7.4 |
| Tesla         |     52 |                6.4 |
| Hyundai       |    191 |                6.2 |
| Land Rover    |     32 |                5.5 |

## Error by age (years)

| age_bucket   |   rows |   median error (%) |
|:-------------|-------:|-------------------:|
| [15, 30)     |    406 |               13.1 |
| [10, 15)     |    825 |               12.8 |
| [6, 10)      |   1234 |                9   |
| [3, 6)       |   1117 |                7.7 |
| [0, 3)       |    909 |                6.5 |

## Error by state

| sellerState   |   rows |   median error (%) |
|:--------------|-------:|-------------------:|
| NM            |     21 |               14.7 |
| IA            |     31 |               12.9 |
| NJ            |    134 |               11.5 |
| ME            |     26 |               11.1 |
| CO            |     89 |               10.4 |
| AR            |     26 |               10.4 |
| MS            |     32 |               10.1 |
| KS            |     27 |                9.8 |
| NC            |    156 |                9.8 |
| CA            |    378 |                9.3 |
| IL            |    134 |                9.3 |
| IN            |     85 |                9.2 |
| FL            |    354 |                9.1 |
| AL            |     58 |                9.1 |
| WA            |     98 |                9   |
| UT            |     35 |                9   |
| NV            |     34 |                8.8 |
| NH            |     34 |                8.8 |
| LA            |     42 |                8.8 |
| NE            |     20 |                8.7 |
| TN            |     98 |                8.6 |
| MA            |    153 |                8.6 |
| TX            |    629 |                8.5 |
| VA            |    145 |                8.4 |
| WI            |     46 |                8.4 |
| SC            |     77 |                8.3 |
| OK            |     80 |                8.2 |
| GA            |    223 |                8.2 |
| PA            |    139 |                8.1 |
| MO            |     67 |                8.1 |
| OH            |    235 |                8.1 |
| MD            |     59 |                8   |
| AZ            |    121 |                8   |
| WV            |     22 |                8   |
| NY            |    179 |                7.9 |
| MN            |     79 |                7.8 |
| KY            |     50 |                7.8 |
| CT            |     41 |                7.5 |
| ID            |     34 |                7.4 |
| MI            |     81 |                7.2 |
| OR            |     54 |                5.5 |

## 20 worst predictions

|    |   year | make          | model                    | trim                     |   mileage | sellerState   |   price |   pred |   pct_err |
|---:|-------:|:--------------|:-------------------------|:-------------------------|----------:|:--------------|--------:|-------:|----------:|
|  0 |   2007 | Ford          | Mustang Shelby GT500     | Coupe RWD                |      2531 | SC            |    1558 |   5830 |     274.2 |
|  1 |   2009 | Ford          | F-350 Super Duty Chassis | nan                      |     18664 | NJ            |    1773 |   5193 |     192.9 |
|  2 |   2008 | Mercedes-Benz | CL-Class                 | CL 550                   |     50212 | NM            |    2822 |   7434 |     163.4 |
|  3 |   2006 | Mercedes-Benz | SLK                      | 280                      |     85834 | TX            |    2657 |   6389 |     140.4 |
|  4 |   2013 | Volkswagen    | Beetle                   | 2.5L with Sunroof        |    135000 | NY            |    1971 |   4559 |     131.3 |
|  5 |   2012 | Aston Martin  | Virage                   | Coupe RWD                |     21991 | NJ            |    3008 |   6951 |     131.1 |
|  6 |   2018 | Bentley       | Bentayga                 | W12 Activity Edition AWD |     20837 | OH            |    6917 |  15921 |     130.2 |
|  7 |   2010 | Toyota        | Corolla                  | Base                     |    115860 | TN            |    1832 |   4122 |     125   |
|  8 |   2009 | Honda         | CR-V                     | LX FWD                   |     96106 | TN            |    1594 |   3561 |     123.4 |
|  9 |   2020 | Ford          | F-150                    | nan                      |    104066 | TX            |    7146 |  15871 |     122.1 |
| 10 |   2008 | Dodge         | RAM 3500                 | ST                       |     78845 | SC            |    2757 |   6099 |     121.2 |
| 11 |   2006 | Dodge         | RAM 2500                 | SLT                      |    144707 | ID            |    2265 |   4982 |     120   |
| 12 |   2006 | Hummer        | H3                       | 4dr SUV 4WD              |    186155 | VA            |    1607 |   3528 |     119.5 |
| 13 |   2011 | Nissan        | Murano                   | S AWD                    |    131347 | FL            |    2435 |   5218 |     114.3 |
| 14 |   2006 | Acura         | MDX                      | AWD with Touring Package |    130351 | WA            |    2053 |   4373 |     113   |
| 15 |   2008 | Chevrolet     | Corvette                 | Coupe RWD                |     28839 | NC            |    1859 |   3886 |     109   |
| 16 |   2010 | Nissan        | Xterra                   | SE                       |    180401 | OH            |    1975 |   4124 |     108.8 |
| 17 |   2010 | Dodge         | RAM 2500                 | Laramie                  |    185784 | FL            |    3986 |   8267 |     107.4 |
| 18 |   2007 | Lincoln       | MKX                      | AWD                      |    139149 | FL            |    2048 |   4227 |     106.4 |
| 19 |   2016 | RAM           | 2500                     | Tradesman                |    224581 | OK            |    5053 |  10331 |     104.4 |

## Coverage

151 make/model combos have >= 30 training rows (used for rounds). 579 are excluded.
