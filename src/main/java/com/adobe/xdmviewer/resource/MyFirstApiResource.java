/*************************************************************************
 * ADOBE CONFIDENTIAL ___________________
 * &lt;p/&gt;
 * Copyright 2019 Adobe Systems Incorporated All Rights Reserved.
 * &lt;p/&gt;
 * NOTICE: All information contained herein is, and remains the property of Adobe Systems
 * Incorporated and its suppliers, if any. The intellectual and technical concepts contained herein
 * are proprietary to Adobe Systems Incorporated and its suppliers and are protected by all
 * applicable intellectual property laws, including trade secret and copyright laws. Dissemination
 * of this information or reproduction of this material is strictly forbidden unless prior written
 * permission is obtained from Adobe Systems Incorporated.
 **************************************************************************/

package com.adobe.xdmviewer.resource;

import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import io.swagger.annotations.ApiResponse;
import io.swagger.annotations.ApiResponses;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import com.adobe.xdmviewer.dto.Something;
import com.adobe.xdmviewer.config.MyFirstApiResourceProperties;
import com.adobe.asr.exception.dto.ErrorResponse;
import com.adobe.asr.telemetry.TelemetryRegistry;
import com.adobe.asr.connector.ims.IMSConnector;

@RestController
@RequestMapping("/xdmviewer")
@Api(value="/myfirstapi")
public class MyFirstApiResource {

  private static Logger logger = LoggerFactory.getLogger(MyFirstApiResource.class);

  @Autowired
  MyFirstApiResourceProperties props;

  @Autowired
  TelemetryRegistry telemetryRegistry;

  @Autowired
  IMSConnector imsConnector;

  @RequestMapping(value = "/myfirstapi",method = RequestMethod.GET)
  @ApiOperation(value = "Displays a message", notes = "The message is read from the properties", response = Something.class )
  @ApiResponses(value = { @ApiResponse(code = 500, message = "Something went wrong", response=ErrorResponse.class) })
  public Something saySomething() {
      logger.info("This is a log in saySomething");
      return new Something("At times something is " + props.getMessage());
      }
}
