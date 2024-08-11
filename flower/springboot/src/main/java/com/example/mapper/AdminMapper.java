package com.example.mapper;

import com.example.entity.Admin;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Select;

public interface AdminMapper {



    @Select("select *from admin where username = #{username}")
    Admin selectByUsername(String username);

    @Insert("insert into admin (username,password,name)"+"values (#{username},#{password},#{name})")
    void insert(Admin admin);
}
