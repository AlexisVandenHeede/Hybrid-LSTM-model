import numpy as np
import pandas as pd
import scipy as s
import matplotlib.pyplot as plt


class ECM():
    def __init__(self, r, p1, p2, p3, q1, q2, q3, battery_num):
        self.r = r
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.battery_num = battery_num

        # load interpolant data 
        self.dat = []
        self.dat.append(pd.read_excel('ECM/SOC_OCV_data.xlsx'))
        self.dat = pd.concat(self.dat)
        self.soc_dat = self.dat['SOC']
        self.ocv_dat = self.dat['OCV']

        # Load interpolant data
        self.interpolants = []
        self.interpolants.append(pd.read_excel('ECM/battery_model.xlsx'))
        self.interpolants = pd.concat(self.interpolants)
        self.SOC_dat = self.interpolants['SOC']
        self.R0_dat = self.interpolants['R0']
        self.R1_dat = self.interpolants['R1']
        self.R2_dat = self.interpolants['R2']
        self.C1_dat = self.interpolants['C1']
        self.C2_dat = self.interpolants['C2']
        self.T_dat = self.interpolants['T']

        # Load battery data
        self.bat = []
        self.bat.append(pd.read_csv('data/' + self.battery_num + '_TTD.csv'))
        self.bat = pd.concat(self.bat)

    def get_discharge_cycles(self):
        ttd = self.bat['TTD']
        indx = []
        for i in range(len(ttd)):
            if ttd[i] == 0:
                indx.append(i)
        indx = np.array(indx)
        return indx
      
    def EKF(self, with_discharge_cycles=False):
        # Initial conditions
        soc_init = 1
        X = np.matrix([[soc_init], [0], [0]])
        deltaT = 1
        Qn_rated = 2.3*3600
        current = self.bat['Current_measured']
        temperature = self.bat['Temperature_measured']
        Vt_act = self.bat['Voltage_measured']

        sococv = np.polyfit(self.soc_dat, self.ocv_dat, 9)
        sococv = np.poly1d(sococv)
        dsococv = np.polyder(sococv)

        # State space model
        R_x = self.r
        P_x = np.matrix([[self.p1, 0, 0], [0, self.p2, 0], [0, 0, self.p3]])
        Q_x = np.matrix([[self.q1, 0, 0], [0, self.q2, 0], [0, 0, self.q3]])

        SOC_est = []
        Vt_est = []
        Vt_err = []

        # EKF
        if with_discharge_cycles:
            idx = self.get_discharge_cycles()
            print(idx)
            for i in range(len(idx)-1):
                print(f'iteration {i}')
                temperature = self.bat['Temperature_measured'][idx[i]:idx[i+1]]
                current = self.bat['Current_measured'][idx[i]:idx[i+1]]
                Vt_act = self.bat['Voltage_measured'][idx[i]:idx[i+1]]

                for k in range(len(current)):
                    T_val = temperature[k+idx[i]]
                    U = current[k+idx[i]]
                    soc = X.item(0)
                    V1 = X[1]
                    V2 = X[2]

                    R0 = s.interpolate.griddata((self.SOC_dat, self.T_dat), self.R0_dat, (soc, T_val), method='nearest')
                    R1 = s.interpolate.griddata((self.SOC_dat, self.T_dat), self.R1_dat, (soc, T_val), method='nearest')
                    R2 = s.interpolate.griddata((self.SOC_dat, self.T_dat), self.R2_dat, (soc, T_val), method='nearest')
                    C1 = s.interpolate.griddata((self.SOC_dat, self.T_dat), self.C1_dat, (soc, T_val), method='nearest')
                    C2 = s.interpolate.griddata((self.SOC_dat, self.T_dat), self.C2_dat, (soc, T_val), method='nearest')

                    ocv_pred = sococv(np.array(soc))

                    tau1 = R1*C1
                    tau2 = R2*C2

                    a1 = np.exp(-deltaT/tau1).item(0)
                    a2 = np.exp(-deltaT/tau2).item(0)

                    b1 = (R1 * (1-a1)).item(0)
                    b2 = (R2 * (1-a2)).item(0)

                    terminal_voltage = ocv_pred - R0 * U - V1 - V2 

                    dOCV = dsococv(soc)

                    C_x = np.matrix([dOCV, -1, -1])

                    Error_x = Vt_act[k+idx[i]] - terminal_voltage

                    Vt_est.append(terminal_voltage.item(0))
                    Vt_err.append(Error_x.item(0))
                    SOC_est.append(soc)

                    # Prediction
                    A = np.matrix([[1, 0, 0], [0, a1, 0], [0, 0, a2]])
                    B = np.matrix([[(-deltaT/Qn_rated)], [b1], [b2]])
                    X = A*X + B*U
                    P_x = A*P_x*A.T + Q_x
                    Kalman_gain = P_x*C_x.T*(C_x*P_x*C_x.T + R_x).I
                    X = X + Kalman_gain*Error_x
                    P_x = (np.eye(3) - Kalman_gain*C_x)*P_x

            total_err = (np.sum(np.square(Vt_err)))/len(Vt_err)
            print(f'MSE is {total_err}')
            return SOC_est, Vt_est, Vt_err, total_err
        else:
            for k in range(len(current)):
                T_val = temperature[k]
                U = current[k]
                soc = X.item(0)
                V1 = X[1]
                V2 = X[2]

                R0 = s.interpolate.griddata((self.SOC_dat, self.T_dat), self.R0_dat, (soc, T_val), method='nearest')
                R1 = s.interpolate.griddata((self.SOC_dat, self.T_dat), self.R1_dat, (soc, T_val), method='nearest')
                R2 = s.interpolate.griddata((self.SOC_dat, self.T_dat), self.R2_dat, (soc, T_val), method='nearest')
                C1 = s.interpolate.griddata((self.SOC_dat, self.T_dat), self.C1_dat, (soc, T_val), method='nearest')
                C2 = s.interpolate.griddata((self.SOC_dat, self.T_dat), self.C2_dat, (soc, T_val), method='nearest')

                ocv_pred = sococv(np.array(soc))

                tau1 = R1*C1
                tau2 = R2*C2

                a1 = np.exp(-deltaT/tau1).item(0)
                a2 = np.exp(-deltaT/tau2).item(0)

                b1 = (R1 * (1-a1)).item(0)
                b2 = (R2 * (1-a2)).item(0)

                terminal_voltage = ocv_pred - R0 * U - V1 - V2 

                dOCV = dsococv(soc)

                C_x = np.matrix([dOCV, -1, -1])

                Error_x = Vt_act[k] - terminal_voltage

                Vt_est.append(terminal_voltage.item(0))
                Vt_err.append(Error_x.item(0))
                SOC_est.append(soc)

                # Prediction
                A = np.matrix([[1, 0, 0], [0, a1, 0], [0, 0, a2]])
                B = np.matrix([[(-deltaT/Qn_rated)], [b1], [b2]])
                X = A*X + B*U
                P_x = A*P_x*A.T + Q_x
                Kalman_gain = P_x*C_x.T*(C_x*P_x*C_x.T + R_x).I
                X = X + Kalman_gain*Error_x
                P_x = (np.eye(3) - Kalman_gain*C_x)*P_x

            total_err = (np.sum(np.square(Vt_err)))/len(Vt_err)
            print(f'MSE is {total_err}')
            return SOC_est, Vt_est, Vt_err, total_err


# testing if the algorithm works
# please uncomment this if you want to run the optimiser
ecm = ECM(0.25098788, 0.3372615, 0.003931529, 0.819609, 0.003931529, 0.8196096, 0.64706235, 'B0005')
soc_est, vt_est, vt_err, total_err = ecm.EKF(with_discharge_cycles=False)
plt.figure(1)
plt.plot(vt_err, label='Error')
plt.legend()
plt.figure(2)
plt.plot(soc_est, label='Estimated SOC')
plt.legend()
plt.show()

# ### old stuff
# def EKF(r, p1, p2, p3, q1, q2, q3, battery_num):
#     # load actual data from file
#     bat = []
#     bat.append(pd.read_csv('data/' + battery_num + '_TTD.csv'))
#     bat = pd.concat(bat)
#     current = bat['Current_measured'][170:354]
#     temperature = bat['Temperature_measured'][170:354]
#     Vt_act = bat['Voltage_measured'][170:354]

#     # Load soc ocv data
#     dat = []
#     dat.append(pd.read_excel('ECM/SOC_OCV_data.xlsx'))
#     dat = pd.concat(dat)
#     soc_dat = dat['SOC']
#     ocv_dat = dat['OCV']

#     # Load interpolant data
#     interpolants = []
#     interpolants.append(pd.read_excel('ECM/battery_model.xlsx'))
#     interpolants = pd.concat(interpolants)
#     SOC_dat = interpolants['SOC']
#     R0_dat = interpolants['R0']
#     R1_dat = interpolants['R1']
#     R2_dat = interpolants['R2']
#     C1_dat = interpolants['C1']
#     C2_dat = interpolants['C2']
#     T_dat = interpolants['T']

#     # Load current and temperature data
#     soc_init = 1
#     X = np.matrix([[soc_init], [0], [0]])
#     deltaT = 1
#     Qn_rated = 2.3*3600

#     sococv = np.polyfit(soc_dat, ocv_dat, 9)
#     sococv = np.poly1d(sococv)
#     dsococv = np.polyder(sococv)

#     # State space model
#     R_x = r
#     P_x = np.matrix([[p1, 0, 0], [0, p2, 0], [0, 0, p3]])
#     Q_x = np.matrix([[q1, 0, 0], [0, q2, 0], [0, 0, q3]])

#     SOC_est = []
#     Vt_est = []
#     Vt_err = []
#     ik = len(current)

#     # EKF
#     for k in range(ik):
#         T_val = temperature[k+170]
#         U = current[k+170]
#         soc = X.item(0)
#         V1 = X[1]
#         V2 = X[2]

#         R0 = s.interpolate.griddata((SOC_dat, T_dat), R0_dat, (soc, T_val), method='nearest')
#         R1 = s.interpolate.griddata((SOC_dat, T_dat), R1_dat, (soc, T_val), method='nearest')
#         R2 = s.interpolate.griddata((SOC_dat, T_dat), R2_dat, (soc, T_val), method='nearest')
#         C1 = s.interpolate.griddata((SOC_dat, T_dat), C1_dat, (soc, T_val), method='nearest')
#         C2 = s.interpolate.griddata((SOC_dat, T_dat), C2_dat, (soc, T_val), method='nearest')

#         ocv_pred = sococv(np.array(soc))

#         tau1 = R1*C1
#         tau2 = R2*C2

#         a1 = np.exp(-deltaT/tau1).item(0)
#         a2 = np.exp(-deltaT/tau2).item(0)

#         b1 = (R1 * (1-a1)).item(0)
#         b2 = (R2 * (1-a2)).item(0)

#         terminal_voltage = ocv_pred - R0 * U - V1 - V2 

#         dOCV = dsococv(soc)

#         C_x = np.matrix([dOCV, -1, -1])

#         Error_x = Vt_act[k+170] - terminal_voltage

#         Vt_est.append(terminal_voltage.item(0))
#         Vt_err.append(Error_x.item(0))
#         SOC_est.append(soc)

#         # Prediction
#         A = np.matrix([[1, 0, 0], [0, a1, 0], [0, 0, a2]])
#         B = np.matrix([[(-deltaT/Qn_rated)], [b1], [b2]])
#         X = A*X + B*U
#         P_x = A*P_x*A.T + Q_x
#         Kalman_gain = P_x*C_x.T*(C_x*P_x*C_x.T + R_x).I
#         X = X + Kalman_gain*Error_x
#         P_x = (np.eye(3) - Kalman_gain*C_x)*P_x

#     total_err = (np.sum(np.square(Vt_err)))/len(Vt_err)
#     print(f'MSE is {total_err}')
#     return SOC_est, Vt_est, Vt_err, Vt_act, total_err
