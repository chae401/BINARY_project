package com.hotdesking.study.web;

import com.hotdesking.study.domain.User;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class CmmnController {
//    @GetMapping("/home")
//    public String home(Model model){
//        //model.addAttribute("data","어서오세요!");
//        return "index";
//    }
    @GetMapping("/join")
    public String join(){
        return "join";
    }

//    @GetMapping("/confirmation")
//    public String confirmation() {return "confirmation";}

    @GetMapping("/find")
    public String find() {return "find";}

    @GetMapping("/login")
    public String login(){return "login";}


}
