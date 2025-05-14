class PIDController:
    def __init__(self, kp, ki, kd, dt, setpoint=0.0, output_limits=(-1, 1)):
        self.kp = kp  # 比例系数
        self.ki = ki  # 积分系数
        self.kd = kd  # 微分系数
        self.dt = dt  # 环境更新时间步长
        self.setpoint = setpoint  # 目标值
        self.output_limits = output_limits # 用来以限制carla中转向角的输出

        # 初始化误差和积分项
        self._previous_error = 0.0
        self._integral = 0.0

    def update(self, current_value):
        """
        计算控制输出。
        :param current_value: 当前测量值
        :param dt: 时间间隔 (秒)
        :return: 控制输出值
        """
        # 计算误差
        error = self.setpoint - current_value

        # 计算比例项
        p_term = self.kp * error

        # 计算积分项
        self._integral += error * self.dt
        i_term = self.ki * self._integral

        # 计算微分项
        derivative = (error - self._previous_error) / self.dt
        d_term = self.kd * derivative

        # 更新先前的误差
        self._previous_error = error

        # 计算总输出
        output = p_term + i_term + d_term
        # 限制输出在指定范围内
        min_output, max_output = self.output_limits
        if min_output is not None:
            output = max(min_output, output)
        if max_output is not None:
            output = min(max_output, output)
        return output

    def set_setpoint(self, setpoint):
        """
        更新目标值。
        :param setpoint: 新的目标值
        """
        self.setpoint = setpoint
        self._integral = 0.0  # 重置积分项，避免突然的积分累积
        self._previous_error = 0.0

# 使用示例
if __name__ == "__main__":
    import time

    # 创建一个 PID 控制器实例
    pid = PIDController(kp=10.0, ki=0.1, kd=0.05, setpoint=10.0)

    # 初始测量值
    current_value = 0.0

    # 模拟控制器执行
    for i in range(100):
        # 时间步长 (假设每次循环间隔 0.1 秒)
        dt = 0.1

        # 更新控制器
        control = pid.update(current_value, dt)

        # 假设控制器作用改变了系统状态（简单起见，直接用控制输出影响当前值）
        current_value += control * dt

        # 输出当前状态
        print(f"Step {i}, Control: {control:.2f}, Current Value: {current_value:.2f}")

        # 等待 0.1 秒 (模拟真实物理系统中的时间步长)
        time.sleep(dt)
