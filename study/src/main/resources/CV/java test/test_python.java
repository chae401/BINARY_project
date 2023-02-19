package com.call.python;

import org.apache.commons.exec.CommandLine;
import org.apache.commons.exec.DefaultExecutor;
import org.apache.commons.exec.PumpStreamHandler;

import java.io.ByteArrayOutputStream;
import java.io.IOException;

public class CallMain {
    public static void main(String[] args)  {
    	System.out.println("Python Call");
        String[] command = new String[4];
        command[0] = "python";
        command[1] = ".\\Hot_Desking_Recognition.py";
        // 아래 두 개는 필요 없는 듯
        // 우리는 입력 필요 없으니깜
        // command[2] = "10";
        // command[3] = "20";
        try {
            execPython(command);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    public static void execPython(String[] command) throws IOException, InterruptedException {
        CommandLine commandLine = CommandLine.parse(command[0]);
        for (int i = 1, n = command.length; i < n; i++) {
            commandLine.addArgument(command[i]);
        }

        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        PumpStreamHandler pumpStreamHandler = new PumpStreamHandler(outputStream);
        DefaultExecutor executor = new DefaultExecutor();
        executor.setStreamHandler(pumpStreamHandler);
        
        int result = executor.execute(commandLine);
        System.out.println("result: " + result);
        System.out.println("user: " + outputStream.toString());

    }    
        
}