package com.example.controller;

import cn.hutool.core.util.ObjectUtil;
import com.example.common.Result;
import com.example.entity.Admin;
import com.example.entity.Comments;
import com.example.entity.Flower;
import com.example.service.AdminService;
import com.example.service.CommentsService;
import com.example.service.FlowerService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import javax.annotation.Resource;
import java.util.List;

@RestController
public class WebController {
    @Resource
    private AdminService adminService;


    /**
     * 默认请求接口
     */
    @GetMapping("/")
    public Result hello() {
        return Result.success();
    }
    /**
     * 登陆接口
     */
    @PostMapping("/login")
    public Result login(@RequestBody Admin admin) {
        Admin dbAdmin = adminService.login(admin);
        return Result.success(dbAdmin);
    }
    /*
    *注册接口
    *
    * */
    @PostMapping("/register")
    public Result register(@RequestBody Admin admin) {
        if(ObjectUtil.isEmpty(admin.getUsername()) || ObjectUtil.isEmpty(admin.getPassword())){
            return Result.error("账号或密码未填");
        }
        adminService.register(admin);
        return Result.success();
    }

    @Resource
    private FlowerService flowerService;


    @PostMapping("/flower")
    public Result flower(@RequestBody Flower flower) {
        Flower dbFlower = flowerService.getFlowerByName(flower);
        return Result.success(dbFlower);
    }

    @Resource
    private CommentsService commentsService;

    @PostMapping("/comments")
    public Result comments(@RequestBody Comments comments){
        List<Comments> dbComments = commentsService.getCommentsByCategory(comments);
        return Result.success(dbComments);
    }


    @PostMapping("/addcomment")
    public Result addcomment(@RequestBody Comments comments) {
        if(ObjectUtil.isEmpty(comments.getContent())){
            return Result.error("评论内容为空！");
        }
        commentsService.addcomment(comments);
        return Result.success();
    }
}
