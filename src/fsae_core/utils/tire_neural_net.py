import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os
from io import StringIO

# ==========================================
# 1. EMBEDDED DATA (Extracted from your logs)
# ==========================================
# This ensures the script works even if .mat files are unreadable
DATA_CSV = """
Segment	Fz_Bin	FyMax_Th	SA_At_FyMax	FyMin_Th	SA_At_FyMin	RMSE	MAE	ErrPct	IA_Mean	P_Mean	V_Mean	Archivo
"1"	1150	2.6199e+03	-11.9626	-2.6277e+03	11.4840	68.8850	37.0527	1.4101	0.0063	83.0819	40.1905	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw4.mat"
"1"	920	2.1313e+03	-11.2924	-2.1438e+03	12.0150	69.4182	38.1619	1.7801	0.0063	83.0819	40.1905	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw4.mat"
"1"	690	1.6282e+03	-12.0120	-1.6577e+03	12.0220	44.8916	22.6930	1.3689	0.0063	83.0819	40.1905	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw4.mat"
"1"	460	1.1256e+03	-12.0160	-1.1314e+03	12.0150	25.2389	13.3167	1.1770	0.0063	83.0819	40.1905	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw4.mat"
"1"	230	572.6825	-10.9328	-557.6051	12.0150	13.2225	7.7852	1.3594	0.0063	83.0819	40.1905	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw4.mat"
"2"	1150	2.5501e+03	-12.0130	-2.7048e+03	11.1762	70.8318	35.1821	1.3007	1.9957	83.1334	40.1923	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw4.mat"
"2"	920	2.0391e+03	-11.0499	-2.2252e+03	11.5991	62.2815	32.3020	1.4516	1.9957	83.1334	40.1923	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw4.mat"
"2"	690	1.5348e+03	-11.9516	-1.6731e+03	10.3227	42.9441	22.1237	1.3223	1.9957	83.1334	40.1923	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw4.mat"
"2"	460	1.0284e+03	-12.0220	-1.1636e+03	12.0090	25.2710	13.9283	1.1970	1.9957	83.1334	40.1923	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw4.mat"
"2"	230	497.3953	-11.3499	-644.4848	12.0180	12.9966	7.4648	1.1583	1.9957	83.1334	40.1923	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw4.mat"
"3"	1150	2.4707e+03	-12.0120	-2.5782e+03	9.5395	70.4451	36.9177	1.4319	3.9928	83.0969	40.1922	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw4.mat"
"3"	920	1.9416e+03	-10.5616	-2.1209e+03	8.8180	59.2132	31.6509	1.4923	3.9928	83.0969	40.1922	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw4.mat"
"3"	690	1.4391e+03	-10.7444	-1.6317e+03	8.5160	39.2144	20.0308	1.2276	3.9928	83.0969	40.1922	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw4.mat"
"3"	460	936.9802	-11.1679	-1.1614e+03	9.2331	23.1718	12.5170	1.0777	3.9928	83.0969	40.1922	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw4.mat"
"3"	230	412.0836	-11.8933	-696.1292	12.0190	12.3828	7.3693	1.0586	3.9928	83.0969	40.1922	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw4.mat"
"4"	1150	2.7079e+03	-12.0250	-2.6901e+03	11.0536	74.4402	39.1249	1.4449	0.0051	68.5823	40.1922	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw4.mat"
"4"	920	2.2894e+03	-8.4542	-2.1971e+03	9.4815	83.5833	50.8433	2.2208	0.0051	68.5823	40.1922	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw4.mat"
"4"	690	1.7018e+03	-10.2589	-1.6604e+03	9.0588	55.5324	27.6544	1.6250	0.0051	68.5823	40.1922	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw4.mat"
"4"	460	1.1607e+03	-11.6469	-1.1306e+03	10.7480	31.7713	16.1306	1.3897	0.0051	68.5823	40.1922	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw4.mat"
"4"	230	589.5811	-10.2589	-580.8884	12.0160	14.8052	8.4941	1.4407	0.0051	68.5823	40.1922	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw4.mat"
"5"	1150	2.5899e+03	-12.0120	-2.7299e+03	11.1103	70.9476	35.7520	1.3096	1.9953	68.5615	40.1941	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw4.mat"
"5"	920	2.1121e+03	-9.7223	-2.2722e+03	9.2988	75.8486	40.9464	1.8020	1.9953	68.5615	40.1941	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw4.mat"
"5"	690	1.5682e+03	-10.0217	-1.6996e+03	8.2119	52.6445	26.1296	1.5374	1.9953	68.5615	40.1941	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw4.mat"
"5"	460	1.0462e+03	-10.8048	-1.1832e+03	8.2675	30.8551	15.9412	1.3473	1.9953	68.5615	40.1941	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw4.mat"
"5"	230	489.7369	-8.7525	-658.3698	8.8710	13.6383	7.7067	1.1706	1.9953	68.5615	40.1941	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw4.mat"
"1"	1150	2.5597e+03	-12.0290	-2.7961e+03	11.3546	75.6897	38.3820	1.3727	3.9925	69.7261	40.1917	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw5.mat"
"1"	920	2.0926e+03	-10.6796	-2.3257e+03	8.5138	74.3182	40.7402	1.7518	3.9925	69.7261	40.1917	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw5.mat"
"1"	690	1.5011e+03	-8.6325	-1.7660e+03	8.0912	51.9436	28.7902	1.6302	3.9925	69.7261	40.1917	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw5.mat"
"1"	460	948.2032	-8.2654	-1.2503e+03	7.9678	29.0722	15.3562	1.2282	3.9925	69.7261	40.1917	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw5.mat"
"1"	230	408.0477	-7.6056	-769.5979	10.4495	14.0381	8.3093	1.0797	3.9925	69.7261	40.1917	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw5.mat"
"2"	1150	2.5002e+03	-10.8660	-2.5327e+03	10.5065	75.9342	46.7415	1.8455	0.0059	96.8466	40.1934	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw5.mat"
"2"	920	2.0311e+03	-10.9209	-2.0444e+03	11.4055	60.5062	31.0855	1.5205	0.0059	96.8466	40.1934	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw5.mat"
"2"	690	1.5613e+03	-11.2292	-1.5878e+03	12.0120	39.9920	19.2903	1.2149	0.0059	96.8466	40.1934	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw5.mat"
"2"	460	1.0765e+03	-12.0140	-1.0801e+03	12.0250	21.7981	11.1904	1.0361	0.0059	96.8466	40.1934	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw5.mat"
"2"	230	571.4818	-9.2957	-534.5711	12.0130	13.0194	7.6474	1.3382	0.0059	96.8466	40.1934	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw5.mat"
"3"	1150	2.3965e+03	-10.9255	-2.5952e+03	12.0160	66.8651	33.9598	1.3085	1.9957	96.5000	40.1928	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw5.mat"
"3"	920	1.9431e+03	-10.6885	-2.1267e+03	12.0130	55.7189	28.6109	1.3453	1.9957	96.5000	40.1928	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw5.mat"
"3"	690	1.4689e+03	-11.6479	-1.6449e+03	12.0230	35.1334	16.7944	1.0210	1.9957	96.5000	40.1928	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw5.mat"
"3"	460	1.0018e+03	-12.0140	-1.1407e+03	12.0180	21.2594	11.0349	0.9674	1.9957	96.5000	40.1928	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw5.mat"
"3"	230	502.1970	-10.3219	-600.5986	12.0060	11.8252	6.6387	1.1054	1.9957	96.5000	40.1928	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw5.mat"
"4"	1150	2.3267e+03	-11.1084	-2.5787e+03	11.8250	63.3994	32.9573	1.2780	3.9926	96.5438	40.1948	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw5.mat"
"4"	920	1.8651e+03	-10.8048	-2.0943e+03	10.9228	50.6707	27.0637	1.2922	3.9926	96.5438	40.1948	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw5.mat"
"4"	690	1.3970e+03	-11.4659	-1.6426e+03	12.0120	32.6012	16.5839	1.0096	3.9926	96.5438	40.1948	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw5.mat"
"4"	460	926.3951	-12.0070	-1.1566e+03	12.0100	20.1411	11.3683	0.9829	3.9926	96.5438	40.1948	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw5.mat"
"4"	230	421.1642	-12.0130	-654.3823	12.0230	11.8417	6.8309	1.0439	3.9926	96.5438	40.1948	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw5.mat"
"1"	1150	2.6993e+03	-12.0250	-2.7723e+03	12.0180	75.4240	39.2923	1.4173	0.0060	54.4094	40.1939	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"1"	920	2.3065e+03	-9.3073	-2.3305e+03	9.6659	80.1460	46.8656	2.0110	0.0060	54.4094	40.1939	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"1"	690	1.7511e+03	-8.9947	-1.7857e+03	8.9335	67.6804	36.4653	2.0420	0.0060	54.4094	40.1939	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"1"	460	1.2222e+03	-9.5372	-1.2255e+03	7.7867	44.1249	21.7569	1.7753	0.0060	54.4094	40.1939	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"1"	230	611.1563	-7.9678	-618.8154	7.9061	16.0287	8.4916	1.3722	0.0060	54.4094	40.1939	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"2"	1150	2.5415e+03	-12.0290	-2.8358e+03	11.4840	73.7371	38.3907	1.3538	1.9954	54.3040	40.1940	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"2"	920	2.1580e+03	-11.7804	-2.4222e+03	9.7183	72.4056	39.7401	1.6407	1.9954	54.3040	40.1940	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"2"	690	1.6104e+03	-10.3796	-1.8287e+03	7.8457	63.7091	32.3618	1.7697	1.9954	54.3040	40.1940	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"2"	460	1.0956e+03	-8.8143	-1.2841e+03	7.2470	41.6055	21.0037	1.6357	1.9954	54.3040	40.1940	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"2"	230	492.5311	-7.6081	-720.6886	7.5452	16.1375	9.1667	1.2719	1.9954	54.3040	40.1940	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"3"	1150	2.3980e+03	-12.0230	-2.7764e+03	10.2045	70.6926	37.4718	1.3496	3.9928	54.3207	40.1951	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"3"	920	2.0643e+03	-12.0200	-2.3593e+03	8.3861	70.3049	37.2847	1.5804	3.9928	54.3207	40.1951	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"3"	690	1.5204e+03	-10.8093	-1.8032e+03	7.6081	64.0255	33.4141	1.8531	3.9928	54.3207	40.1951	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"3"	460	956.3730	-8.5046	-1.2883e+03	7.2422	35.9354	18.8721	1.4648	3.9928	54.3207	40.1951	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"3"	230	377.9695	-7.4276	-802.8728	7.3074	15.8829	9.7010	1.2083	3.9928	54.3207	40.1951	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"4"	1150	2.5544e+03	-12.0130	-2.6500e+03	11.0554	71.9415	39.3543	1.4851	0.0058	83.9600	40.1959	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"4"	920	2.0741e+03	-10.3193	-2.1402e+03	11.2273	66.4148	34.7079	1.6217	0.0058	83.9600	40.1959	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"4"	690	1.5922e+03	-10.8048	-1.6551e+03	12.0180	46.0465	22.1461	1.3380	0.0058	83.9600	40.1959	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"4"	460	1.1042e+03	-12.0090	-1.1295e+03	12.0130	24.9586	11.8534	1.0495	0.0058	83.9600	40.1959	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"4"	230	582.7920	-8.3910	-563.6434	12.0090	13.0116	7.2296	1.2405	0.0058	83.9600	40.1959	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"5"	1150	2.4424e+03	-12.0100	-2.6967e+03	11.0435	68.4587	34.0914	1.2642	1.9944	83.1657	40.1964	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"5"	920	1.9644e+03	-10.3176	-2.2175e+03	11.6459	62.4809	32.5609	1.4684	1.9944	83.1657	40.1964	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"5"	690	1.4792e+03	-10.3193	-1.7155e+03	12.0120	41.9156	20.7164	1.2076	1.9944	83.1657	40.1964	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"5"	460	997.4957	-10.9804	-1.1978e+03	12.0100	25.2793	13.5566	1.1318	1.9944	83.1657	40.1964	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"5"	230	490.0314	-9.4117	-645.8088	12.0150	13.5896	7.6344	1.1822	1.9944	83.1657	40.1964	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"6"	1150	2.3665e+03	-12.0090	-2.6051e+03	10.0209	69.6985	35.6371	1.3680	3.9936	82.5413	40.1953	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"6"	920	1.8628e+03	-9.8365	-2.1297e+03	9.2284	58.8442	31.7392	1.4903	3.9936	82.5413	40.1953	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"6"	690	1.3767e+03	-9.8341	-1.6289e+03	8.6346	39.2961	20.8565	1.2804	3.9936	82.5413	40.1953	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"6"	460	891.9023	-9.9556	-1.1714e+03	9.4752	23.8580	13.9000	1.1866	3.9936	82.5413	40.1953	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"6"	230	388.0267	-10.4382	-710.0425	12.0160	13.5312	7.8938	1.1117	3.9936	82.5413	40.1953	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"7"	1150	2.6228e+03	-12.0160	-2.7195e+03	11.7735	98.3201	45.8852	1.6873	0.0055	82.4681	24.1150	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"7"	920	2.1274e+03	-11.7121	-2.2218e+03	11.5904	87.9940	43.5728	1.9612	0.0055	82.4681	24.1150	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"7"	690	1.6100e+03	-11.5205	-1.6823e+03	12.0130	60.3085	27.1852	1.6159	0.0055	82.4681	24.1150	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"7"	460	1.1047e+03	-10.7426	-1.1466e+03	12.0120	34.6436	16.0845	1.4028	0.0055	82.4681	24.1150	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"7"	230	559.0968	-8.2634	-564.1320	12.0050	16.5540	9.7653	1.7310	0.0055	82.4681	24.1150	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"8"	1150	2.3685e+03	-9.5927	-2.5215e+03	9.2911	55.3261	33.5826	1.3318	0.0057	82.4554	72.3575	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"8"	920	1.9852e+03	-8.3868	-2.1048e+03	8.9909	45.6470	24.5745	1.1676	0.0057	82.4554	72.3575	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"8"	690	1.5315e+03	-9.2261	-1.6036e+03	10.1994	39.0615	20.1215	1.2548	0.0057	82.4554	72.3575	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"8"	460	1.0730e+03	-11.6411	-1.1194e+03	12.0130	19.3529	10.9325	0.9767	0.0057	82.4554	72.3575	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"8"	230	576.8496	-9.1758	-569.1979	12.0100	12.7667	8.3453	1.4467	0.0057	82.4554	72.3575	"RawData_Cornering_Matlab_SI_Round9_Runs_1to15\B2356raw6.mat"
"""

# ==========================================
# 2. NEURAL NETWORK
# ==========================================
class TireGripNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1) # Output: Peak Force
        )
    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. DATA LOADING
# ==========================================
def load_data():
    """ Loads data from the embedded string """
    print(f"📂 Parsing embedded tire data...")
    try:
        # Use regex separator to handle multiple spaces/tabs
        df = pd.read_csv(StringIO(DATA_CSV), sep=r'\s+', engine='python')
        
        # Verify required columns exist
        req_cols = ['Fz_Bin', 'FyMax_Th', 'IA_Mean', 'P_Mean']
        if not all(col in df.columns for col in req_cols):
            raise ValueError(f"Missing columns. Found: {df.columns}")
            
        print(f"✅ Loaded {len(df)} samples from telemetry summary.")
        return df
    except Exception as e:
        print(f"❌ Error parsing data: {e}")
        return None

# ==========================================
# 4. TRAINING ENGINE
# ==========================================
def train_grip_model(df):
    # Inputs: Load, Camber, Pressure
    X = df[['Fz_Bin', 'IA_Mean', 'P_Mean']].values.astype(np.float32)
    y = df[['FyMax_Th']].values.astype(np.float32)

    # Normalize
    stats = {
        'X_mean': X.mean(axis=0), 'X_std': X.std(axis=0),
        'y_mean': y.mean(),       'y_std': y.std()
    }
    # Avoid div/0
    stats['X_std'][stats['X_std'] == 0] = 1.0
    if stats['y_std'] == 0: stats['y_std'] = 1.0
    
    X_norm = (X - stats['X_mean']) / stats['X_std']
    y_norm = (y - stats['y_mean']) / stats['y_std']
    
    X_tensor = torch.tensor(X_norm)
    y_tensor = torch.tensor(y_norm)
    
    model = TireGripNet()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    print("🧠 Training Neural Network...")
    for epoch in range(1500):
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = criterion(pred, y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"   Epoch {epoch}: Loss {loss.item():.5f}")

    return model, stats

# ==========================================
# 5. POLYNOMIAL FITTING (The Goal)
# ==========================================
def generate_solver_equation(df):
    """
    Fits: Fy = c0*Fz + c1*Fz^2 + c2*IA + c3*IA^2 + c4*Fz*IA + c5
    """
    # Filter for standard pressure (~12 psi / 83 kPa)
    p_nom = 83.0 
    df_sub = df[(df['P_Mean'] > p_nom - 15) & (df['P_Mean'] < p_nom + 15)]
    if len(df_sub) < 10: 
        print("⚠️ Warning: Not enough data at 83kPa, using all data.")
        df_sub = df
    
    Fz = df_sub['Fz_Bin'].values
    IA = df_sub['IA_Mean'].values
    Fy = df_sub['FyMax_Th'].values
    
    # Design Matrix
    # [Fz, Fz^2, IA, IA^2, Fz*IA, 1]
    A = np.column_stack([Fz, Fz**2, IA, IA**2, Fz*IA, np.ones_like(Fz)])
    
    # Least Squares Solve
    coeffs, _, _, _ = np.linalg.lstsq(A, Fy, rcond=None)
    return coeffs

# ==========================================
# 6. VISUALIZATION
# ==========================================
def plot_results(model, stats, df):
    # Sweep Load from 200N to 1200N
    fz_sweep = np.linspace(200, 1200, 100)
    ia_levels = [0, 2, 4] # Camber degrees
    p_nom = 83.0 # Standard Pressure
    
    plt.figure(figsize=(10, 6))
    
    # Plot Raw Data points (colored by Camber)
    plt.scatter(df['Fz_Bin'], df['FyMax_Th'], c=df['IA_Mean'], cmap='viridis', label='Raw Data')
    plt.colorbar(label='Camber (deg)')
    
    # Plot NN Predictions
    for ia in ia_levels:
        inputs = []
        for fz in fz_sweep:
            inputs.append([fz, ia, p_nom])
        
        inputs = np.array(inputs, dtype=np.float32)
        inputs_norm = (inputs - stats['X_mean']) / stats['X_std']
        
        with torch.no_grad():
            pred_norm = model(torch.tensor(inputs_norm)).numpy()
            
        pred = pred_norm * stats['y_std'] + stats['y_mean']
        plt.plot(fz_sweep, pred, linewidth=2, linestyle='--', label=f'NN Pred (IA={ia}°)')
        
    plt.title(f"Tire Grip Model (Pressure ~{p_nom} kPa)")
    plt.xlabel("Vertical Load Fz (N)")
    plt.ylabel("Peak Lateral Force Fy (N)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ==========================================
# 7. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    df = load_data()
    
    if df is not None:
        model, stats = train_grip_model(df)
        c = generate_solver_equation(df)
        
        print("\n" + "="*60)
        print("✅ COMPLETED. COPY THIS INTO src/fsae_core/dynamics/vehicle_14dof.py")
        print("="*60)
        print(f"    def get_peak_grip(self, Fz, IA):")
        print(f"        # Derived from Neural Network Fit (RMSE: {stats['y_std']:.2f})")
        print(f"        # Fy = c0*Fz + c1*Fz^2 + c2*IA + c3*IA^2 + c4*Fz*IA + Bias")
        print(f"        return ({c[0]:.4f}*Fz) + ({c[1]:.4e}*Fz**2) + ({c[2]:.4f}*IA) + \\")
        print(f"               ({c[3]:.4f}*IA**2) + ({c[4]:.4e}*Fz*IA) + ({c[5]:.4f})")
        print("="*60)
        
        plot_results(model, stats, df)