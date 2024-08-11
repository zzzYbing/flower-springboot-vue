package com.example.mapper;

import com.example.entity.Comments;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Select;

import java.util.List;

public interface CommentsMapper {

    @Select("select *from comments where category = #{category}")
    List<Comments> selectCategory(String category);

    @Insert("insert into comments (category,user,time,content)"+"values (#{category},#{user},#{time},#{content})")
    void insertcomments(Comments comments);
}
