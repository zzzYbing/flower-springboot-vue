package com.example.service;



import cn.hutool.core.util.ObjectUtil;
import com.example.entity.Comments;
import com.example.mapper.CommentsMapper;
import org.springframework.stereotype.Service;

import javax.annotation.Resource;
import java.util.Date;
import java.util.List;

@Service
public class CommentsService {
    @Resource
    private CommentsMapper commentsMapper;

    public List<Comments> getCommentsByCategory(Comments comments){
        return commentsMapper.selectCategory(comments.getCategory());
    }

    //新增评论
    public void addcomment(Comments comments) {
        Comments dbComments = new Comments();
        dbComments.setContent(comments.getContent()); //内容
        dbComments.setCategory(comments.getCategory());//种类
        dbComments.setUser(comments.getUser());//用户


        Date currentDate = new Date();

        dbComments.setTime(currentDate); // 设置时间
        commentsMapper.insertcomments(dbComments);
    }
}
