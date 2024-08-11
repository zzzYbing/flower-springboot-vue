package com.example.mapper;

import com.example.entity.Flower;
import org.apache.ibatis.annotations.Select;

public interface FlowerMapper {

    @Select("select *from flower where name = #{name}")
    Flower selectName(String name);
}
