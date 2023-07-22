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
        self.bat.append(pd.read_csv(f'data/padded_data_data_[{self.battery_num}].csv'))
        self.bat = pd.concat(self.bat)

    def get_discharge_cycles(self):
        ttd = self.bat['TTD']
        indx = [2]
        for i in range(len(ttd)-10):
            if ttd[i] == 0 and ttd[i+10] == 0:
                indx.append(i+2+10)
        indx = np.array(indx)
        return indx

    def EKF(self, save_plot=False, save_data=False):
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

        # EKF
        idx = self.get_discharge_cycles()
        total_err = 0
        self.main_data_soc = []
        self.main_data_vt_est = []
        for i in range(len(idx)-1):
            SOC_est = []
            Vt_est = []
            Vt_err = []
            temperature = self.bat['Temperature_measured'][idx[i]+1:idx[i+1]-1]
            current = self.bat['Current_measured'][idx[i]+1:idx[i+1]-1]
            Vt_act = self.bat['Voltage_measured'][idx[i]+1:idx[i+1]-1]
            indx = []
            Vt_act_plot = []

            for k in range(len(current)):
                T_val = temperature[k+idx[i]+1]
                U = current[k+idx[i]+1]
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
                Error_x = Vt_act[k+idx[i]+1] - terminal_voltage

                Vt_est.append(terminal_voltage.item(0))
                Vt_err.append(Error_x.item(0))
                SOC_est.append(soc)
                indx.append(k+idx[i]+2)
                Vt_act_plot.append(Vt_act[k+idx[i]+1])

                # Prediction
                A = np.matrix([[1, 0, 0], [0, a1, 0], [0, 0, a2]])
                B = np.matrix([[(-deltaT/Qn_rated)], [b1], [b2]])
                X = A*X + B*U
                P_x = A*P_x*A.T + Q_x
                Kalman_gain = P_x*C_x.T*(C_x*P_x*C_x.T + R_x).I
                X = X + Kalman_gain*Error_x
                P_x = (np.eye(3) - Kalman_gain*C_x)*P_x
            
            # remove extra data points
            Vt_est = Vt_est[10:]
            Vt_err = Vt_err[10:]
            SOC_est = SOC_est[10:]
            indx = indx[10:]
            Vt_act_plot = Vt_act_plot[10:]
            total_err += np.sum(np.square(Vt_err))

            if save_plot:
                # print(Vt_est)
                plt.figure(1)
                plt.plot(indx, Vt_act_plot, 'r')
                plt.plot(indx, Vt_est, 'b')
                plt.xlabel('Instance')
                plt.ylabel('Terminal voltage')
                plt.figure(2)
                plt.plot(indx, Vt_err, 'g')
                plt.xlabel('Instance')
                plt.ylabel('Absolute error')
                plt.figure(3)
                plt.plot(indx, SOC_est, 'b')
                plt.xlabel('Instance')
                plt.ylabel('SOC')
            
            if save_data:
                for p in range(len(SOC_est)):
                    self.main_data_soc.append(SOC_est[p])
                    self.main_data_vt_est.append(Vt_est[p])

        plt.show()
        total_err = total_err/(len(idx)-1)
        print(f'MSE is {total_err}')
        return SOC_est, Vt_est, Vt_err, total_err
    
    def save_data(self):
        df_soc = pd.DataFrame(self.main_data_soc)
        df_soc.to_csv(f'ECM/soc_est_{self.battery_num}.csv')
        df_vt = pd.DataFrame(self.main_data_vt_est)
        df_vt.to_csv(f'ECM/vt_est_{self.battery_num}.csv')


# testing if the algorithm works
# please uncomment this if you want to run the optimiser
# ecm = ECM(0.25098788, 0.3372615, 0.003931529, 0.819609, 0.003931529, 0.8196096, 0.64706235, 'B0005')  # higher MSE but clear rul degredation wo. discharge cycles
# # # ecm = ECM(3.2156930588235295, 5.098044117647058, 0.4313821176470588, 9.490196588235294, 0.4313821176470588, 9.490196588235294, 8.35294282352941, 'B0005')  # Weird spiking behaviour no real rul degredation wo. discharge cycles
ecm = ECM(988.3529, 88.628588, 96.0788235, 40.39811, 96.0788235, 40.39811, 0.40211, 'B0005')  # ga opt values w. discharge cycles
# ecm = ECM(r=2.078439294117647, p1=8.901961882352941, p2=0.11765694117647059, p3=4.23529988235294, q1=0.11765694117647059, q2=4.23529988235294, q3=9.137255764705882, battery_num='B0005')
# ecm = ECM(r=7.647061176470587, p1=8.745099294117647, p2=2.4313801176470586, p3=6.862748235294117, q1=2.4313801176470586, q2=6.862748235294117, q3=7.607845529411764, battery_num='B0006')
soc_est, vt_est, vt_err, total_err = ecm.EKF(save_plot=True, save_data=True)
ecm.save_data()
