# Price model report

> Trained on real **CARFAX One Price** value estimates (Rebrowser Carfax sample).

## Test-set accuracy

| metric | our model |
|---|---|
| rows | 4306 |
| MAE ($) | 2305.0 |
| MAPE (%) | 6.9 |
| median error (%) | 4.4 |
| within 10% (%) | 80.0 |
| within 20% (%) | 93.8 |

Target median error <= 10%: **MET**

Note: some error is unavoidable. Options, condition and local demand aren't in the features, so two cars that look identical here can be worth different amounts.

## Error by make

| make          |   rows |   median error (%) |
|:--------------|-------:|-------------------:|
| Porsche       |     55 |                9.6 |
| Mercedes-Benz |    184 |                9.3 |
| Audi          |     84 |                8.8 |
| Land Rover    |     29 |                6.9 |
| Dodge         |    124 |                6.6 |
| Genesis       |     28 |                6.6 |
| Volvo         |     28 |                6.4 |
| Lincoln       |     34 |                6.4 |
| Cadillac      |     57 |                6   |
| Chrysler      |     65 |                5.7 |
| Buick         |     33 |                5.3 |
| BMW           |    255 |                5.1 |
| Chevrolet     |    504 |                5.1 |
| Tesla         |     33 |                4.9 |
| Lexus         |    118 |                4.8 |
| Ford          |    586 |                4.5 |
| Ram           |    176 |                4.3 |
| Kia           |    110 |                4.2 |
| Jeep          |    187 |                3.9 |
| GMC           |    147 |                3.9 |
| Nissan        |    199 |                3.7 |
| Toyota        |    448 |                3.3 |
| Volkswagen    |    105 |                3.3 |
| Hyundai       |    127 |                3.1 |
| Subaru        |    113 |                3   |
| Mazda         |     72 |                2.9 |
| Honda         |    272 |                2.6 |
| Acura         |     46 |                2.3 |

## Error by age (years)

| age_bucket   |   rows |   median error (%) |
|:-------------|-------:|-------------------:|
| [15, 30)     |    236 |               10.3 |
| [10, 15)     |    672 |                4.9 |
| [6, 10)      |    869 |                4.7 |
| [3, 6)       |   1195 |                4.5 |
| [0, 3)       |   1334 |                3.6 |

## Error by state

| sellerState   |   rows |   median error (%) |
|:--------------|-------:|-------------------:|
| UT            |     21 |                9.5 |
| SC            |     51 |                6.2 |
| NM            |     42 |                5.6 |
| OR            |     59 |                5.6 |
| LA            |     25 |                5.3 |
| FL            |    410 |                5.1 |
| MN            |     82 |                5.1 |
| NV            |     60 |                5.1 |
| NJ            |    203 |                5   |
| WA            |     83 |                4.9 |
| KS            |     25 |                4.9 |
| AZ            |    204 |                4.8 |
| CO            |     91 |                4.8 |
| GA            |    210 |                4.8 |
| WV            |     22 |                4.7 |
| IL            |    174 |                4.7 |
| TN            |    119 |                4.7 |
| MO            |     68 |                4.6 |
| MI            |    138 |                4.6 |
| CT            |     40 |                4.6 |
| PA            |    159 |                4.5 |
| NH            |     21 |                4.5 |
| KY            |     53 |                4.4 |
| IN            |     93 |                4.4 |
| AR            |     47 |                4.3 |
| OK            |     85 |                4.3 |
| NC            |    155 |                4.2 |
| WI            |     95 |                4.2 |
| OH            |    177 |                4.1 |
| VA            |    180 |                4.1 |
| AL            |    102 |                4.1 |
| CA            |    486 |                4   |
| IA            |     44 |                3.9 |
| MA            |     70 |                3.8 |
| NY            |    171 |                3.7 |
| ID            |     40 |                3.6 |
| MS            |     46 |                3.3 |
| MD            |     97 |                3.1 |

## 20 worst predictions

|    |   year | make          | model      | trim          |   mileage | sellerState   |   price |   pred |   pct_err |
|---:|-------:|:--------------|:-----------|:--------------|----------:|:--------------|--------:|-------:|----------:|
|  0 |   2006 | Ford          | F-150      | STX           |    197602 | IN            |    2700 |   6008 |     122.5 |
|  1 |   2000 | Porsche       | Boxster    | nan           |    136300 | SC            |    7580 |  15978 |     110.8 |
|  2 |   2012 | Ford          | Fusion     | SE            |    170000 | KY            |    2970 |   6057 |     103.9 |
|  3 |   2002 | Mercedes-Benz | S-Class    | S 430         |    117386 | WA            |    6120 |  11640 |      90.2 |
|  4 |   2002 | Jaguar        | X-Type     | nan           |     94091 | GA            |    5580 |  10207 |      82.9 |
|  5 |   2007 | Chrysler      | PT Cruiser | Touring       |     25267 | KY            |    8030 |  14317 |      78.3 |
|  6 |   2000 | Mercedes-Benz | CLK        | 320           |    111169 | OH            |    5090 |   9019 |      77.2 |
|  7 |   2011 | Lamborghini   | Gallardo   | LP560         |     17955 | NC            |  146250 |  37732 |      74.2 |
|  8 |   2013 | Fiat          | 500        | Lounge        |     30727 | WA            |   10070 |  17209 |      70.9 |
|  9 |   2010 | Rolls-Royce   | Ghost      | nan           |     38844 | TN            |   80480 |  23571 |      70.7 |
| 10 |   2020 | Mercedes-Benz | E-Class    | E 450         |     25606 | PA            |   40350 |  68097 |      68.8 |
| 11 |   2017 | Audi          | R8         | 5.2           |     48737 | OH            |  137760 |  49070 |      64.4 |
| 12 |   2017 | Acura         | NSX        | nan           |      6548 | CA            |  140300 |  58041 |      58.6 |
| 13 |   2000 | BMW           | Z3         | 2.3           |    203041 | MO            |    3690 |   5814 |      57.6 |
| 14 |   2013 | Kia           | Soul       | Base          |     28210 | OH            |    7240 |  11392 |      57.4 |
| 15 |   2013 | Mercedes-Benz | S-Class    | S 550         |     59051 | FL            |   15370 |  24014 |      56.2 |
| 16 |   2005 | Mercedes-Benz | C-Class    | C 240         |     86035 | UT            |    6420 |   9919 |      54.5 |
| 17 |   2012 | GMC           | Savana     | 1500 Upfitter |    256491 | MI            |   15370 |   7454 |      51.5 |
| 18 |   2014 | BMW           | 7 Series   | 750Li         |     89447 | CA            |   14160 |  21352 |      50.8 |
| 19 |   2025 | Volvo         | V60        | B5 Plus       |     20049 | CA            |   35630 |  53677 |      50.7 |

## Coverage

143 make/model combos have >= 30 training rows (used for rounds). 356 are excluded.

## Independent check: Kelley Blue Book

8,963 used AutoTrader cars in models we cover (history unknown, so this is a rough check).

| metric | value |
|---|---|
| our price inside KBB fair range | 20.2% |
| within KBB range ±10% | 53.8% |
| median our / KBB midpoint | 1.044 |
| median abs. gap to KBB midpoint | 14.3% |

By make (≥30 cars):

| make       |   cars |   our / KBB mid |
|:-----------|-------:|----------------:|
| Cadillac   |     65 |           0.884 |
| GMC        |    501 |           0.894 |
| Dodge      |    226 |           0.913 |
| Volvo      |     36 |           0.943 |
| BMW        |    253 |           0.965 |
| Toyota     |   1212 |           0.981 |
| Chevrolet  |   1421 |           1.001 |
| Jeep       |    811 |           1.01  |
| Ford       |    921 |           1.014 |
| Tesla      |     42 |           1.016 |
| Subaru     |    363 |           1.026 |
| Land Rover |     73 |           1.041 |
| Honda      |    873 |           1.064 |
| Porsche    |    123 |           1.107 |
| Hyundai    |    524 |           1.125 |
| Kia        |    227 |           1.13  |
| Volkswagen |    281 |           1.135 |
| Audi       |    119 |           1.151 |
| Chrysler   |     83 |           1.171 |
| Nissan     |    579 |           1.178 |
| Acura      |    116 |           1.193 |
| Buick      |     68 |           1.204 |
