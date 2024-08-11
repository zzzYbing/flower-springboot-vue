package com.example.service;


import com.example.entity.Flower;
import com.example.mapper.FlowerMapper;
import org.springframework.stereotype.Service;

import javax.annotation.Resource;

@Service
public class FlowerService {
    @Resource
    private FlowerMapper flowerMapper;

    public Flower getFlowerByName(Flower flower){
        Flower dbFlower = flowerMapper.selectName(flower.getName());

        return  dbFlower;
    }
}
