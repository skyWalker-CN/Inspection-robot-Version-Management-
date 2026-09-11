def update_odometry(self, left_speed, right_speed):
        """更新里程计信息"""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time
        
        # 计算线速度和角速度
        v = (left_speed + right_speed) / 2.0
        w = (right_speed - left_speed) / self.wheel_separation
        
        # 更新位置
        delta_x = v * math.cos(self.th) * dt
        delta_y = v * math.sin(self.th) * dt
        delta_th = w * dt
        
        with self.lock:  # 保护共享变量
            self.x += delta_x
            self.y += delta_y
            self.th += delta_th
        
        # 发布里程计信息
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"
        
        # 设置位置
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        
        # 设置方向 (转换为四元数)
        q = quaternion_from_euler(0, 0, self.th)
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        
        # 设置速度
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = w
        
        self.odom_pub.publish(odom)

        # 发布TF变换
        t = TransformStamped()
        t.header.stamp = current_time.to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id = "base_footprint"
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)
