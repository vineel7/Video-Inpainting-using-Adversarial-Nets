/*************************************************************************
 * ADOBE CONFIDENTIAL ___________________
 * <p/>
 * Copyright 2019 Adobe Systems Incorporated All Rights Reserved.
 * <p/>
 * NOTICE: All information contained herein is, and remains the property of Adobe Systems
 * Incorporated and its suppliers, if any. The intellectual and technical concepts contained herein
 * are proprietary to Adobe Systems Incorporated and its suppliers and are protected by all
 * applicable intellectual property laws, including trade secret and copyright laws. Dissemination
 * of this information or reproduction of this material is strictly forbidden unless prior written
 * permission is obtained from Adobe Systems Incorporated.
 **************************************************************************/

package com.adobe.xdmviewer.resource;

import org.junit.Test;
import org.junit.Before;

import static org.assertj.core.api.Assertions.assertThat;

import org.mockito.Mock;
import org.mockito.InjectMocks;
import org.mockito.MockitoAnnotations;

import static org.mockito.Mockito.when;


import com.adobe.xdmviewer.config.MyFirstApiResourceProperties;

/**
 * Sample Unit Test for MyFirstApiResource.
 * The test creates a mock object for MyFirstApiResourceProperties, injects mocks for MyFirstApiResource
 * and invokes methods on mocked object to test MyFirstApiResource.
 */

public class MyFirstApiResourceTest {

	@Mock
	private MyFirstApiResourceProperties propsMock;

	@InjectMocks
	private MyFirstApiResource apiMock;

	@Before
	public void setUp() {
		MockitoAnnotations.initMocks(this);
	}

	@Test
	public void testSaySomething() {
	    String message = "Everything";
		when(propsMock.getMessage()).thenReturn(message);
		assertThat(apiMock.saySomething().getMessage()).isEqualTo("At times something is "+message);
	}
}
