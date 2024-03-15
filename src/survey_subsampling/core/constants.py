#!/usr/bin/env python

CBCL_ABCL_cannot_be_harmonized = ["cl1", "cl2", "cl6", "cl15", "cl23", "cl30", "cl38", "cl44",
                                  "cl47", "cl49", "cl53", "cl55", "cl56h", "cl59", "cl60",
                                  "cl64", "cl67", "cl72", "cl73", "cl76", "cl78", "cl81",
                                  "cl83", "cl89", "cl92", "cl96", "cl98", "cl99", "cl106",
                                  "cl107", "cl108", "cl109", "cl110", "cl113"]

CBCL_items = ['cl1', 'cl2', 'cl3', 'cl4', 'cl5', 'cl6', 'cl7', 'cl8', 'cl9', 'cl10', 'cl11',
              'cl12', 'cl13', 'cl14', 'cl15', 'cl16', 'cl17', 'cl18', 'cl19', 'cl20', 'cl21',
              'cl22', 'cl23', 'cl24', 'cl25', 'cl26', 'cl27', 'cl28', 'cl29', 'cl30', 'cl31',
              'cl32', 'cl33', 'cl34', 'cl35', 'cl36', 'cl37', 'cl38', 'cl39', 'cl40', 'cl41',
              'cl42', 'cl43', 'cl44', 'cl45', 'cl46', 'cl47', 'cl48', 'cl49', 'cl50', 'cl51',
              'cl52', 'cl53', 'cl54', 'cl55', 'cl56', 'cl56a', 'cl56b', 'cl56c', 'cl56d',
              'cl56e', 'cl56f', 'cl56g', 'cl56h', 'cl57', 'cl58', 'cl59', 'cl60', 'cl61',
              'cl62', 'cl63', 'cl64', 'cl65', 'cl66', 'cl67', 'cl68', 'cl69', 'cl70', 'cl71',
              'cl72', 'cl73', 'cl74', 'cl75', 'cl76', 'cl77', 'cl78', 'cl79', 'cl80', 'cl81',
              'cl82', 'cl83', 'cl84', 'cl85', 'cl86', 'cl87', 'cl88', 'cl89', 'cl90', 'cl91',
              'cl92', 'cl93', 'cl94', 'cl95', 'cl96', 'cl97', 'cl98', 'cl99', 'cl100', 'cl101',
              'cl102', 'cl103', 'cl104', 'cl105', 'cl106', 'cl107', 'cl108', 'cl109', 'cl110',
              'cl111', 'cl112', 'cl113']

ABCL_items = ['al1', 'al2', 'al3', 'al4', 'al5', 'al6', 'al7', 'al8', 'al9', 'al10', 'al11',
              'al12', 'al13', 'al14', 'al15', 'al16', 'al17', 'al18', 'al19', 'al20', 'al21',
              'al22', 'al23', 'al24', 'al25', 'al26', 'al27', 'al28', 'al29', 'al30', 'al31',
              'al32', 'al33', 'al34', 'al35', 'al36', 'al37', 'al38', 'al39', 'al40', 'al41',
              'al42', 'al43', 'al44', 'al45', 'al46', 'al47', 'al48', 'al49', 'al50', 'al51',
              'al52', 'al53', 'al54', 'al55', 'al56', 'al56a', 'al56b', 'al56c', 'al56d',
              'al56e', 'al56f', 'al56g', 'al57', 'al58', 'al59', 'al60', 'al61', 'al62',
              'al63', 'al64', 'al65', 'al66', 'al67', 'al68', 'al69', 'al70', 'al71', 'al72',
              'al73', 'al74', 'al75', 'al76', 'al77', 'al78', 'al79', 'al80', 'al81', 'al82',
              'al83', 'al84', 'al85', 'al86', 'al87', 'al88', 'al89', 'al90', 'al91', 'al92',
              'al93', 'al94', 'al95', 'al96', 'al97', 'al98', 'al99', 'al100', 'al101', 'al102',
              'al103', 'al104', 'al105', 'al106', 'al107', 'al108', 'al109', 'al110', 'al111',
              'al112', 'al113', 'al114', 'al115', 'al116', 'al117', 'al118', 'al119', 'al120',
              'al121', 'al122', 'al123', 'al124', 'al125', 'al126']

CBCLABCL_items = list(set(CBCL_items) - set(CBCL_ABCL_cannot_be_harmonized))

Dx_labels_all = ["dcany", "dcanyanx", "dcanydep", "dcanyhk", "dcanycd", "dcsepa",
                 "dcspph", "dcsoph", "dcpanic", "dcagor", "dcptsd", "dcocd",
                 "dcgena", "dcdmdd", "dcmadep", "dcmania", "dcodd", "dccd"]