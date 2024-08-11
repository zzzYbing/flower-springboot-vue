package com.example.service;

import cn.hutool.core.util.ObjectUtil;
import com.example.entity.Admin;
import com.example.exception.CustomException;
import com.example.mapper.AdminMapper;
import org.springframework.stereotype.Service;

import javax.annotation.Resource;

@Service
public class AdminService {


    @Resource
    private AdminMapper adminMapper;
    /**
     *
     * 登陆
     */

    public Admin login(Admin admin){
        Admin dbAdmin = adminMapper.selectByUsername(admin.getUsername());
        if(dbAdmin == null){
            //没有用户
            throw new CustomException("账号或密码错误");
        }
        //比较密码
        if (!admin.getPassword().equals(dbAdmin.getPassword())){
            throw new CustomException("账号或密码错误");
        }
        //登陆成功
        return dbAdmin;
    }

    //注册
    public void register(Admin admin) {
        Admin dbAdmin = new Admin();
        dbAdmin.setUsername(admin.getUsername()); //账号
        dbAdmin.setPassword(admin.getPassword());//密码
        this.add(dbAdmin);
    }

    //新增
    private void add(Admin admin) {
        Admin dbAdmin = adminMapper.selectByUsername(admin.getUsername());
        if(dbAdmin != null){//已有同名账号，不允许插入
                throw new CustomException("账号已存在");
        }
        if(ObjectUtil.isEmpty(admin.getName())){
            admin.setName(admin.getUsername());
        }
        adminMapper.insert(admin);
    }
}
