% 定义传递函数
s = tf('s');  % 拉普拉斯变量 s
Phi_s = 0.276*s^2;  % Phi(s)
U_s = s^4+0.11*s^3-33.06*s^2-2.694*s;  % U(s)
 
P_bai = Phi_s / U_s;  % 传递函数
 
% pidtune 函数自动调整 PID 参数
[Kpid, ~, ~] = pidtune(P_bai, 'PID');
 
% 显示 PID 控制器的参数
disp('PID 控制器参数：');
disp(Kpid);
 
% 构建闭环控制系统
closed_loop_system = feedback(Kpid * P_bai, 1);
 
% 仿真闭环系统的bode图
figure;
bode(closed_loop_system);
title('闭环系统的bode图');
grid on;
 
% 仿真闭环系统的单位阶跃响应
figure;
step(closed_loop_system);
title('闭环系统的单位阶跃响应');
grid on;
 
% 仿真闭环系统的单位脉冲响应
figure;
impulse(closed_loop_system);
title('闭环系统的单位脉冲响应');
grid on;
