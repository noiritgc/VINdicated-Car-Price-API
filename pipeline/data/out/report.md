# Price model report

> Trained on real **CARFAX One Price** value estimates (Rebrowser Carfax sample).

## Test-set accuracy

| metric | our model |
|---|---|
| rows | 4306 |
| MAE ($) | 2285.0 |
| MAPE (%) | 6.9 |
| median error (%) | 4.5 |
| within 10% (%) | 80.5 |
| within 20% (%) | 94.0 |

Target median error <= 10%: **MET**

Note: some error is unavoidable. Options, condition and local demand aren't in the features, so two cars that look identical here can be worth different amounts.

## Error by make

| make          |   rows |   median error (%) |
|:--------------|-------:|-------------------:|
| Porsche       |     55 |               10.8 |
| Mercedes-Benz |    184 |                8.6 |
| Audi          |     84 |                8.5 |
| Genesis       |     28 |                7.8 |
| Volvo         |     28 |                7.3 |
| Dodge         |    124 |                6.6 |
| Land Rover    |     29 |                6.1 |
| Lincoln       |     34 |                6   |
| BMW           |    255 |                5.4 |
| Chrysler      |     65 |                5.3 |
| Cadillac      |     57 |                5.2 |
| Tesla         |     33 |                5.1 |
| Chevrolet     |    504 |                5.1 |
| Ford          |    586 |                4.8 |
| Ram           |    176 |                4.7 |
| Lexus         |    118 |                4.4 |
| Kia           |    110 |                4.3 |
| Jeep          |    187 |                4.3 |
| Buick         |     33 |                4.2 |
| Mazda         |     72 |                4.1 |
| Nissan        |    199 |                3.7 |
| GMC           |    147 |                3.7 |
| Acura         |     46 |                3.6 |
| Subaru        |    113 |                3.3 |
| Volkswagen    |    105 |                3.3 |
| Toyota        |    448 |                3.2 |
| Hyundai       |    127 |                3   |
| Honda         |    272 |                2.5 |

## Error by age (years)

| age_bucket   |   rows |   median error (%) |
|:-------------|-------:|-------------------:|
| [15, 30)     |    236 |               11.2 |
| [10, 15)     |    672 |                4.9 |
| [6, 10)      |    869 |                4.7 |
| [3, 6)       |   1195 |                4.6 |
| [0, 3)       |   1334 |                3.5 |

## Error by state

| sellerState   |   rows |   median error (%) |
|:--------------|-------:|-------------------:|
| UT            |     21 |               10.9 |
| OR            |     59 |                5.7 |
| SC            |     51 |                5.6 |
| NV            |     60 |                5.5 |
| NM            |     42 |                5.4 |
| WA            |     83 |                5.3 |
| FL            |    410 |                5.3 |
| LA            |     25 |                5.3 |
| TN            |    119 |                5.2 |
| CT            |     40 |                5   |
| IL            |    174 |                4.9 |
| OK            |     85 |                4.8 |
| WV            |     22 |                4.8 |
| AR            |     47 |                4.8 |
| CO            |     91 |                4.8 |
| KY            |     53 |                4.7 |
| AZ            |    204 |                4.7 |
| NJ            |    203 |                4.7 |
| NC            |    155 |                4.7 |
| MO            |     68 |                4.6 |
| MN            |     82 |                4.6 |
| IN            |     93 |                4.4 |
| CA            |    486 |                4.4 |
| GA            |    210 |                4.4 |
| NH            |     21 |                4.3 |
| IA            |     44 |                4.3 |
| MI            |    138 |                4.1 |
| PA            |    159 |                4.1 |
| ID            |     40 |                4   |
| WI            |     95 |                3.9 |
| VA            |    180 |                3.9 |
| AL            |    102 |                3.8 |
| MA            |     70 |                3.8 |
| KS            |     25 |                3.7 |
| MD            |     97 |                3.6 |
| OH            |    177 |                3.5 |
| NY            |    171 |                3.5 |
| MS            |     46 |                3.3 |

## 20 worst predictions

|    |   year | make          | model      | trim          |   mileage | sellerState   |   price |   pred |   pct_err |
|---:|-------:|:--------------|:-----------|:--------------|----------:|:--------------|--------:|-------:|----------:|
|  0 |   2006 | Ford          | F-150      | STX           |    197602 | IN            |    2700 |   6140 |     127.4 |
|  1 |   2000 | Porsche       | Boxster    | nan           |    136300 | SC            |    7580 |  15549 |     105.1 |
|  2 |   2002 | Mercedes-Benz | S-Class    | S 430         |    117386 | WA            |    6120 |  12372 |     102.2 |
|  3 |   2012 | Ford          | Fusion     | SE            |    170000 | KY            |    2970 |   5826 |      96.2 |
|  4 |   2002 | Jaguar        | X-Type     | nan           |     94091 | GA            |    5580 |   9775 |      75.2 |
|  5 |   2020 | Mercedes-Benz | E-Class    | E 450         |     25606 | PA            |   40350 |  70516 |      74.8 |
|  6 |   2007 | Chrysler      | PT Cruiser | Touring       |     25267 | KY            |    8030 |  14007 |      74.4 |
|  7 |   2010 | Rolls-Royce   | Ghost      | nan           |     38844 | TN            |   80480 |  20649 |      74.3 |
|  8 |   2011 | Lamborghini   | Gallardo   | LP560         |     17955 | NC            |  146250 |  41314 |      71.8 |
|  9 |   2000 | Mercedes-Benz | CLK        | 320           |    111169 | OH            |    5090 |   8682 |      70.6 |
| 10 |   2005 | Mercedes-Benz | C-Class    | C 240         |     86035 | UT            |    6420 |  10799 |      68.2 |
| 11 |   2017 | Acura         | NSX        | nan           |      6548 | CA            |  140300 |  45564 |      67.5 |
| 12 |   2017 | Audi          | R8         | 5.2           |     48737 | OH            |  137760 |  48927 |      64.5 |
| 13 |   2013 | Fiat          | 500        | Lounge        |     30727 | WA            |   10070 |  16473 |      63.6 |
| 14 |   2025 | Alfa Romeo    | Tonale     | nan           |      4949 | MA            |   29110 |  45904 |      57.7 |
| 15 |   2012 | GMC           | Savana     | 1500 Upfitter |    256491 | MI            |   15370 |   6937 |      54.9 |
| 16 |   2005 | Mercedes-Benz | C-Class    | C 240         |     69024 | IN            |    8500 |  13164 |      54.9 |
| 17 |   2014 | BMW           | 7 Series   | 750Li         |     89447 | CA            |   14160 |  21779 |      53.8 |
| 18 |   2013 | Mercedes-Benz | S-Class    | S 550         |     59051 | FL            |   15370 |  23417 |      52.4 |
| 19 |   2011 | Honda         | CR-V       | SE            |    275677 | VA            |    4430 |   6736 |      52.1 |

## Coverage

143 make/model combos have >= 30 training rows (used for rounds). 356 are excluded.

## Independent check: Kelley Blue Book

8,963 used AutoTrader cars in models we cover (history unknown, so this is a rough check).

| metric | value |
|---|---|
| our price inside KBB fair range | 21.4% |
| within KBB range ±10% | 56.7% |
| median our / KBB midpoint | 1.040 |
| median abs. gap to KBB midpoint | 13.4% |

By make (≥30 cars):

| make       |   cars |   our / KBB mid |
|:-----------|-------:|----------------:|
| Cadillac   |     65 |           0.875 |
| Dodge      |    226 |           0.903 |
| BMW        |    253 |           0.922 |
| Volvo      |     36 |           0.943 |
| GMC        |    501 |           0.947 |
| Tesla      |     42 |           0.999 |
| Jeep       |    811 |           1     |
| Toyota     |   1212 |           1.002 |
| Chevrolet  |   1421 |           1.018 |
| Ford       |    921 |           1.019 |
| Subaru     |    363 |           1.021 |
| Land Rover |     73 |           1.023 |
| Honda      |    873 |           1.029 |
| Audi       |    119 |           1.076 |
| Porsche    |    123 |           1.108 |
| Nissan     |    579 |           1.115 |
| Chrysler   |     83 |           1.118 |
| Hyundai    |    524 |           1.118 |
| Kia        |    227 |           1.136 |
| Volkswagen |    281 |           1.142 |
| Buick      |     68 |           1.178 |
| Acura      |    116 |           1.235 |
