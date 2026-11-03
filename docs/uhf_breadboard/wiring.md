# 배선표 (블록별, 넷마다 연결할 핀)

각 줄은 한 넷입니다: 그 핀들을 모두 한 점(또는 한 선)으로 납땜합니다. `GND`는 접지 버스, 다른 블록의 핀은 *기울임*으로 표시합니다. 전해 커패시터는 핀 1이 +입니다(설계 선택 cp_polarity: 부품의 극성 표시와 대조).

## power - power: 9 V battery, power switch, 1N5817, TX / RX switch, two L78L05

- **VBAT**: J101.1(Pin_1), SW101.2(B)
- **VBAT_ON**: SW101.1(A), D101.2(A)
- **V9**: D101.1(K), SW102.2(B), C101.1(+), C102.1
- **V_RX**: SW102.1(A), U102.3(VI), C105.1, C107.1(+), *R501.1*, *C512.1*, *U501.7(V+)*, *C513.1*, *U502.7(V+)*, *U503.7(V+)*, *C514.1*, *R509.1*, *Q501.2(C)*
- **V_TX**: SW102.3(C), U101.3(VI), C103.1, *R201.1*, *C208.1*, *U201.7(V+)*, *C211.1*, *U202.7(V+)*, *R307.1*, *R311.1*, *R315.1*, *R319.1*, *R601.1*
- **V5T**: U101.1(VO), C104.1, *R203.1*, *R207.1*, *Q301.3(C)*, *R301.1*, *C304.1*
- **V5R**: U102.1(VO), C106.1, *R402.1*, *R404.1*, *R405.1*, *R408.1*
- **GND** (접지 버스): J101.2, U101.2, U102.2, C101.2, C102.2, C103.2, C104.2, C105.2, C106.2, C107.2

## tx_audio - TX audio: microphone amplifier with soft limiter, Sallen-Key low-pass, deviation trimmer, varactor bias

- **V_TX**: *SW102.3(C)*, *U101.3(VI)*, *C103.1*, R201.1, C208.1, U201.7(V+), C211.1, U202.7(V+), *R307.1*, *R311.1*, *R315.1*, *R319.1*, *R601.1*
- **V5T**: *U101.1(VO)*, *C104.1*, R203.1, R207.1, *Q301.3(C)*, *R301.1*, *C304.1*
- **GND** (접지 버스): R202.2, C201.2, MK201.1, C207.2, C208.2, U201.4, C210.2, C211.2, U202.4, RV201.1, R208.2, C204.2
- **VREF_T**: R201.2, R202.1, C201.1(+), R204.2
- **MIC_P**: MK201.2(+), R203.2, C205.2(-)
- **TXA**: D201.2(A), D202.1(K), R209.1, C206.1, U201.6(OUT), R211.1
- **TXN**: D201.1(K), D202.2(A), R209.2, C206.2, R210.1, U201.2(-)
- **TXP**: C205.1(+), R204.1, U201.3(+)
- **TXG**: R210.2, C207.1(+)
- **TXS1**: R211.2, R212.1, C209.1
- **TXS2**: R212.2, C210.1, U202.3(+)
- **TXF**: C209.2, U202.2(-), U202.6(OUT), C203.1(+)
- **DEV_TOP**: C203.2(-), RV201.3
- **DEV_W**: RV201.2, C212.2(-)
- **DEV_C**: C212.1(+), R205.1
- **VAR**: R205.2, R206.2, *Y301.2*, *D301.1(K)*, *CT301.1*
- **VBIAS**: R206.1, R207.2, R208.1, C204.1(+)

## tx_rf - TX RF: crystal VXO with varactor FM, x3 / x2 / x2 multipliers, 3-resonator band-pass, class-A PA, L-match, 5th-order low-pass

- **V_TX**: *SW102.3(C)*, *U101.3(VI)*, *C103.1*, *R201.1*, *C208.1*, *U201.7(V+)*, *C211.1*, *U202.7(V+)*, R307.1, R311.1, R315.1, R319.1, *R601.1*
- **V5T**: *U101.1(VO)*, *C104.1*, *R203.1*, *R207.1*, Q301.3(C), R301.1, C304.1
- **GND** (접지 버스): D301.2, CT301.2, R302.2, C303.2, R303.2, C304.2, C307.2, R305.2, R306.2, C306.2, CT302.2, L302.2, C309.2, R309.2, R310.2, C308.2, CT303.2, L304.2, C311.2, R313.2, R314.2, C310.2, CT304.2, L306.2, CT305.2, L307.2, CT306.2, L308.2, C322.2, R317.2, R318.2, C328.2, C323.2, C325.2, C326.2, C327.2
- **VAR**: *R205.2*, *R206.2*, Y301.2, D301.1(K), CT301.1
- **XB**: Y301.1, Q301.2(B), R301.2, R302.1, C302.1
- **XE**: Q301.1(E), C302.2, C303.1, R303.1, C305.1
- **TX_X3_B**: C305.2, R304.2, R305.1, Q302.3(B)
- **TX_X3_VC**: R307.2, C307.1, R304.1, L301.1
- **TX_X3_E**: R306.1, C306.1, Q302.2(E)
- **TX_T1**: C312.2, C313.1, CT302.1, L302.1
- **TX_X3_C**: Q302.1(C), L301.2, C312.1
- **TX_X6_B**: C313.2, R308.2, R309.1, Q303.1(B)
- **TX_X6_VC**: R311.2, C309.1, R308.1, L303.1
- **TX_X6_E**: R310.1, C308.1, Q303.2(E)
- **TX_T2**: C315.2, C316.1, CT303.1, L304.1
- **TX_X6_C**: Q303.3(C), L303.2, C315.1
- **TX_X12_B**: C316.2, R312.2, R313.1, Q304.1(B)
- **TX_X12_VC**: R315.2, C311.1, R312.1, L305.1
- **TX_X12_E**: R314.1, C310.1, Q304.2(E)
- **TX_B1**: C318.2, CT304.1, L306.1, C319.1
- **TX_B2**: C319.2, CT305.1, L307.1, C320.1
- **TX_B3**: C321.1, C320.2, CT306.1, L308.1
- **TX_X12_C**: Q304.3(C), L305.2, C318.1
- **PA_VC**: R319.2, C322.1, R316.1, L309.1
- **PA_B**: R316.2, R317.1, Q305.1(B), C321.2
- **PA_E**: R318.1, C328.1, Q305.2(E)
- **PA_C**: L309.2, Q305.3(C), C323.1, L310.1
- **PA_M**: L310.2, C324.1
- **LPF_IN**: C324.2, C325.1, L311.1
- **LPF_N1**: L311.2, C326.1, L312.1
- **TX_OUT**: L312.2, C327.1, *D601.2(A)*, *L602.2*

## trx - T/R switch: series 1N4148 to the antenna, lumped lambda/4 to the receiver, shunt 1N4148 at its far end, SMA

- **V_TX**: *SW102.3(C)*, *U101.3(VI)*, *C103.1*, *R201.1*, *C208.1*, *U201.7(V+)*, *C211.1*, *U202.7(V+)*, *R307.1*, *R311.1*, *R315.1*, *R319.1*, R601.1
- **GND** (접지 버스): D602.1, C601.2, C602.2, C603.2, J601.2
- **TX_OUT**: *L312.2*, *C327.1*, D601.2(A), L602.2
- **ANT_N**: D601.1(K), L601.1, C601.1, C604.1
- **RXQ**: D602.2(A), L601.2, C602.1, C605.1
- **ANT**: C604.2, J601.1(In)
- **RX_IN**: C605.2, *Q401.2(E)*, *R401.1*
- **TR_B**: R601.2, L602.1, C603.1

## rx - RX: common-base RF amplifier and self-quenched super-regenerative detector

- **V5R**: *U102.1(VO)*, *C106.1*, R402.1, R404.1, R405.1, R408.1
- **GND** (접지 버스): R401.2, R403.2, C401.2, R406.2, C404.2, C407.2, R407.2, C406.2, C408.2
- **RX_IN**: *C605.2*, Q401.2(E), R401.1
- **RXB**: Q401.1(B), R402.2, R403.1, C401.1
- **RXC**: Q401.3(C), R404.2, C403.1
- **SR_C**: C403.2, Q402.3(C), L401.2, CT401.1, C405.1
- **SR_B**: Q402.1(B), R405.2, R406.1, C404.1
- **SR_E**: Q402.2(E), C405.2, L402.1
- **SR_VT**: L401.1, CT401.2, C407.1, RV401.2, RV401.3
- **SR_Q**: L402.2, R407.1, C406.1, R409.1
- **SR_REG**: R408.2, RV401.1
- **AF_RAW**: R409.2, C408.1, *C502.2(-)*

## rx_audio - RX audio: gain stage, Sallen-Key low-pass, volume, op-amp driven BD139 / BD140 output stage

- **V_RX**: *SW102.1(A)*, *U102.3(VI)*, *C105.1*, *C107.1(+)*, R501.1, C512.1, U501.7(V+), C513.1, U502.7(V+), U503.7(V+), C514.1, R509.1, Q501.2(C)
- **GND** (접지 버스): R502.2, C501.2, C504.2, C512.2, U501.4, C506.2, C513.2, U502.4, RV501.1, RV501.MP, U503.4, C514.2, R510.2, Q502.2, C509.2, C511.2, J501.2
- **AF_RAW**: *R409.2*, *C408.1*, C502.2(-)
- **VREF_R**: R501.2, R502.1, C501.1(+), R503.2, R508.2
- **RXP**: C502.1(+), R503.1, U501.3(+)
- **RXA**: R504.1, C503.1, U501.6(OUT), R506.1
- **RXN**: R504.2, C503.2, R505.1, U501.2(-)
- **RXG**: R505.2, C504.1(+)
- **RXS1**: R506.2, R507.1, C505.1
- **RXS2**: R507.2, C506.1, U502.3(+)
- **RXF**: C505.2, U502.2(-), U502.6(OUT), C507.1(+)
- **VOL_TOP**: C507.2(-), RV501.3
- **VOL_W**: RV501.2, C508.2(-)
- **PA_IN**: C508.1(+), R508.1, U503.3(+)
- **PA_N**: U503.2(-), R513.2, R514.1
- **DRV**: U503.6(OUT), D501.1(K), D502.2(A)
- **BN**: D501.2(A), R509.2, Q501.3(B)
- **BP**: D502.1(K), R510.1, Q502.3(B)
- **E1**: Q501.1(E), R511.1
- **E2**: Q502.1(E), R512.1
- **OUTE**: R511.2, R512.2, R513.1, C510.1(+), R515.1
- **PA_G**: R514.2, C509.1(+)
- **SPK_P**: C510.2(-), J501.1(Pin_1)
- **ZOB**: R515.2, C511.1

## 부품 값

| 부품 | 값 | 설명 |
|---|---|---|
| J101 | 9V | 9 V battery snap (Pin_1 = +) |
| SW101 | POWER | power switch (B common, A = on, C = off) |
| D101 | 1N5817 | reverse-polarity diode in series with the battery |
| SW102 | TX/RX | transmit / receive switch (B common from V9, A = receive -> V_RX, C = transmit -> V_TX) |
| U101 | L78L05 | V5T: the VXO's regulated 5 V (on only while transmitting) |
| U102 | L78L05 | V5R: the receiver's regulated 5 V (on only while receiving) |
| C101 | 100u | V9 bulk |
| C102 | 100n | V9 decoupling |
| C103 | 330n | U1 input capacitor |
| C104 | 100n | V5T decoupling |
| C105 | 330n | U2 input capacitor |
| C106 | 100n | V5R decoupling |
| C107 | 100u | V_RX bulk (the speaker stage) |
| R201 | 100k | VREF_T divider (top) |
| R202 | 100k | VREF_T divider (bottom) |
| C201 | 10u | VREF_T decoupling |
| MK201 | Mic | electret microphone (pin 2 '+' to the bias resistor) |
| R203 | 4.7k | electret bias resistor |
| D201 | 1N4148 | microphone amplifier (TL071, soft limiter): soft limiter diode across R_f |
| D202 | 1N4148 | microphone amplifier (TL071, soft limiter): soft limiter diode across R_f |
| U201 | TL071 | microphone amplifier (TL071, soft limiter) |
| C205 | 1u | microphone amplifier (TL071, soft limiter): input coupling |
| R204 | 100k | microphone amplifier (TL071, soft limiter): bias to the reference |
| R209 | 100k | microphone amplifier (TL071, soft limiter): feedback resistor |
| C206 | 470p | microphone amplifier (TL071, soft limiter): feedback roll-off capacitor |
| R210 | 1k | microphone amplifier (TL071, soft limiter): ground-leg resistor |
| C207 | 10u | microphone amplifier (TL071, soft limiter): ground-leg electrolytic |
| C208 | 100n | microphone amplifier (TL071, soft limiter): supply decoupling |
| U202 | TL071 | TX low-pass (Sallen-Key) |
| R211 | 10k | TX low-pass (Sallen-Key): R1 |
| R212 | 10k | TX low-pass (Sallen-Key): R2 |
| C209 | 7.5026n | TX low-pass (Sallen-Key): C1 (feedback) |
| C210 | 3.7513n | TX low-pass (Sallen-Key): C2 (to ground) |
| C211 | 100n | TX low-pass (Sallen-Key): supply decoupling |
| C203 | 4.7u | coupling into the deviation trimmer |
| RV201 | 10k | deviation trimmer |
| C212 | 1u | deviation coupling (+ on the biased side) |
| R205 | 47k | deviation resistor into the varactor node |
| R206 | 47k | varactor bias |
| R207 | 27k | VBIAS divider (top) |
| R208 | 33k | VBIAS divider (bottom) |
| C204 | 10u | VBIAS decoupling |
| Y301 | 36M | VXO crystal at f_c / 12 = 36 MHz (fundamental mode) |
| D301 | BB910 | VXO varactor in series with the crystal (direct FM) |
| Q301 | 2N3904 | VXO transistor (Colpitts, crystal from base to the varactor) |
| CT301 | 10p | VXO frequency trimmer (across the varactor) |
| R301 | 22k | VXO base divider (top) |
| R302 | 22k | VXO base divider (bottom) |
| C302 | 68p | Colpitts base-emitter capacitor |
| C303 | 68p | Colpitts emitter-ground capacitor |
| R303 | 1k | VXO emitter resistor |
| C304 | 100n | V5T decoupling at the VXO |
| C305 | 22p | VXO output coupling to the tripler |
| Q302 | BF199 | multiplier stage 1 (BF199, x3 to 108 MHz) |
| R307 | 47 | multiplier stage 1 (BF199, x3 to 108 MHz): supply feed resistor (decouples the stage from V_TX) |
| C307 | 1n | multiplier stage 1 (BF199, x3 to 108 MHz): feed decoupling |
| R304 | 47k | multiplier stage 1 (BF199, x3 to 108 MHz): upper base-divider resistor |
| R305 | 4.7k | multiplier stage 1 (BF199, x3 to 108 MHz): lower base-divider resistor |
| R306 | 220 | multiplier stage 1 (BF199, x3 to 108 MHz): emitter resistor |
| C306 | 1n | multiplier stage 1 (BF199, x3 to 108 MHz): emitter bypass |
| L301 | 470n | multiplier stage 1 (BF199, x3 to 108 MHz): collector feed choke |
| C312 | 1.865p | tx_tank1 (single-tuned tank at 108 MHz): input tap capacitor |
| CT302 | 10.432p | tx_tank1 (single-tuned tank at 108 MHz): resonator 1 shunt capacitor |
| C313 | 1.4522p | tx_tank1 (single-tuned tank at 108 MHz): output tap capacitor |
| L302 | 150n | tx_tank1 (single-tuned tank at 108 MHz): resonator 1 inductor |
| Q303 | BFR92A | multiplier stage 2 (BFR92A, x2 to 216 MHz) |
| R311 | 47 | multiplier stage 2 (BFR92A, x2 to 216 MHz): supply feed resistor (decouples the stage from V_TX) |
| C309 | 1n | multiplier stage 2 (BFR92A, x2 to 216 MHz): feed decoupling |
| R308 | 47k | multiplier stage 2 (BFR92A, x2 to 216 MHz): upper base-divider resistor |
| R309 | 4.7k | multiplier stage 2 (BFR92A, x2 to 216 MHz): lower base-divider resistor |
| R310 | 220 | multiplier stage 2 (BFR92A, x2 to 216 MHz): emitter resistor |
| C308 | 1n | multiplier stage 2 (BFR92A, x2 to 216 MHz): emitter bypass |
| L303 | 220n | multiplier stage 2 (BFR92A, x2 to 216 MHz): collector feed choke |
| C315 | 974.57f | tx_tank2 (single-tuned tank at 216 MHz): input tap capacitor |
| CT303 | 5.9072p | tx_tank2 (single-tuned tank at 216 MHz): resonator 1 shunt capacitor |
| C316 | 710.88f | tx_tank2 (single-tuned tank at 216 MHz): output tap capacitor |
| L304 | 68n | tx_tank2 (single-tuned tank at 216 MHz): resonator 1 inductor |
| Q304 | BFR92A | multiplier stage 3 (BFR92A, x2 to 432 MHz) |
| R315 | 47 | multiplier stage 3 (BFR92A, x2 to 432 MHz): supply feed resistor (decouples the stage from V_TX) |
| C311 | 1n | multiplier stage 3 (BFR92A, x2 to 432 MHz): feed decoupling |
| R312 | 47k | multiplier stage 3 (BFR92A, x2 to 432 MHz): upper base-divider resistor |
| R313 | 4.7k | multiplier stage 3 (BFR92A, x2 to 432 MHz): lower base-divider resistor |
| R314 | 330 | multiplier stage 3 (BFR92A, x2 to 432 MHz): emitter resistor |
| C310 | 1n | multiplier stage 3 (BFR92A, x2 to 432 MHz): emitter bypass |
| L305 | 100n | multiplier stage 3 (BFR92A, x2 to 432 MHz): collector feed choke |
| C318 | 650.2f | tx_bpf (the x12 stage's 3-resonator band-pass at f_c): input tap capacitor |
| CT304 | 7.7596p | tx_bpf (the x12 stage's 3-resonator band-pass at f_c): resonator 1 shunt capacitor |
| C319 | 177.73f | tx_bpf (the x12 stage's 3-resonator band-pass at f_c): coupling capacitor between resonators 1 and 2 |
| CT305 | 8.6931p | tx_bpf (the x12 stage's 3-resonator band-pass at f_c): resonator 2 shunt capacitor |
| C320 | 177.73f | tx_bpf (the x12 stage's 3-resonator band-pass at f_c): coupling capacitor between resonators 2 and 3 |
| CT306 | 8.3942p | tx_bpf (the x12 stage's 3-resonator band-pass at f_c): resonator 3 shunt capacitor |
| C321 | 609.24f | tx_bpf (the x12 stage's 3-resonator band-pass at f_c): output tap capacitor |
| L306 | 15n | tx_bpf (the x12 stage's 3-resonator band-pass at f_c): resonator 1 inductor |
| L307 | 15n | tx_bpf (the x12 stage's 3-resonator band-pass at f_c): resonator 2 inductor |
| L308 | 15n | tx_bpf (the x12 stage's 3-resonator band-pass at f_c): resonator 3 inductor |
| Q305 | BFR92A | class-A PA (about 30 mW designed) |
| R319 | 47 | PA supply feed resistor |
| C322 | 1n | PA feed decoupling |
| R316 | 3.3k | PA upper base divider |
| R317 | 1.2k | PA lower base divider |
| R318 | 100 | PA emitter resistor |
| C328 | 1n | PA emitter bypass |
| L309 | 470n | PA collector feed choke |
| C323 | 1.8924p | pa_match: shunt capacitor at the load line |
| L310 | 66.629n | pa_match: series coil to the system impedance |
| C324 | 100p | DC block into the low-pass |
| C325 | 7.605p | lpf element 1 (shunt C) |
| L311 | 22.733n | lpf element 2 (series L) |
| C326 | 13.097p | lpf element 3 (shunt C) |
| L312 | 22.733n | lpf element 4 (series L) |
| C327 | 7.605p | lpf element 5 (shunt C) |
| D601 | 1N4148 | T/R switch: series diode TX_OUT -> antenna (on while transmitting) |
| D602 | 1N4148 | T/R switch: shunt diode at the receiver's end of the lambda/4 (on while transmitting) |
| L601 | 18.421n | lumped lambda/4: series coil |
| C601 | 7.3683p | lumped lambda/4: shunt C at the antenna end |
| C602 | 7.3683p | lumped lambda/4: shunt C at the receiver end |
| C604 | 100p | DC block to the SMA |
| C605 | 100p | DC block to the receiver |
| R601 | 1k | T/R bias resistor |
| L602 | 470n | T/R bias choke onto TX_OUT |
| C603 | 1n | T/R bias bypass |
| J601 | SMA | antenna / 50 ohm dummy-load connector |
| Q401 | BFR92A | RX RF amplifier (common base) |
| Q402 | BFR92A | super-regenerative detector |
| R401 | 1k | RF amplifier emitter resistor |
| R402 | 10k | RF amplifier base divider (top) |
| R403 | 4.7k | RF amplifier base divider (bottom) |
| C401 | 1n | RF amplifier base bypass |
| R404 | 1k | RF amplifier collector load |
| C403 | 1p | coupling into the detector's tank |
| R405 | 10k | detector base divider (top) |
| R406 | 10k | detector base divider (bottom) |
| C404 | 1n | detector base bypass |
| L401 | 22n | detector tank coil |
| CT401 | 6.1695p | detector tuning trimmer (set to the carrier) |
| C405 | 2.2p | detector feedback capacitor |
| C407 | 1n | tank cold-end bypass |
| L402 | 1u | detector emitter choke |
| R407 | 1k | quench resistor |
| C406 | 4.7n | quench capacitor |
| R408 | 47 | regeneration series resistor |
| RV401 | 1k | regeneration trimmer (rheostat) |
| R409 | 10k | quench filter resistor |
| C408 | 4.7n | quench filter capacitor |
| R501 | 100k | VREF_R divider (top) |
| R502 | 100k | VREF_R divider (bottom) |
| C501 | 10u | VREF_R decoupling |
| U501 | TL071 | RX audio amplifier (TL071) |
| C502 | 1u | RX audio amplifier (TL071): input coupling |
| R503 | 100k | RX audio amplifier (TL071): bias to the reference |
| R504 | 100k | RX audio amplifier (TL071): feedback resistor |
| C503 | 470p | RX audio amplifier (TL071): feedback roll-off capacitor |
| R505 | 1k | RX audio amplifier (TL071): ground-leg resistor |
| C504 | 10u | RX audio amplifier (TL071): ground-leg electrolytic |
| C512 | 100n | RX audio amplifier (TL071): supply decoupling |
| U502 | TL071 | RX low-pass (Sallen-Key; also the quench filter) |
| R506 | 10k | RX low-pass (Sallen-Key; also the quench filter): R1 |
| R507 | 10k | RX low-pass (Sallen-Key; also the quench filter): R2 |
| C505 | 7.5026n | RX low-pass (Sallen-Key; also the quench filter): C1 (feedback) |
| C506 | 3.7513n | RX low-pass (Sallen-Key; also the quench filter): C2 (to ground) |
| C513 | 100n | RX low-pass (Sallen-Key; also the quench filter): supply decoupling |
| RV501 | 10k | volume |
| C507 | 4.7u | volume coupling |
| C508 | 1u | output stage input coupling |
| R508 | 100k | output stage input bias |
| U503 | TL071 | output stage driver (TL071, the BD139 / BD140 inside its loop) |
| C514 | 100n | U3 supply decoupling |
| D501 | 1N4148 | output stage bias diode (two drops between the bases) |
| D502 | 1N4148 | output stage bias diode (two drops between the bases) |
| R509 | 1k | upper base feed |
| R510 | 1k | lower base feed |
| Q501 | BD139 | output stage NPN (upper half) |
| Q502 | BD140 | output stage PNP (lower half) |
| R511 | 1 | upper emitter resistor |
| R512 | 1 | lower emitter resistor |
| R513 | 22k | output stage feedback |
| R514 | 4.7k | output stage ground leg |
| C509 | 10u | output stage ground-leg electrolytic |
| C510 | 220u | speaker coupling |
| R515 | 10 | Zobel resistor |
| C511 | 100n | Zobel capacitor |
| J501 | SPK | 8 ohm speaker (Pin_1 = +) |
