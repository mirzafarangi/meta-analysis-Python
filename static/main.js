function renderFixedModelChart (containerId, data) {
  let fixedModelChartData = []

  fixedModelChartData.push({
    "category": "",
    "measure": 0,
    "bullet": "",
    "bulletSize": 0,
    "high": 0,
    "low": 0
  })

  for (var i = 0; i < data.g_list.length; i++) {
    fixedModelChartData.push({
      "category": data.study_list[i],
      "measure": data.g_list[i].toFixed(3),
      "bullet": "square",
      "bulletSize": data.weight_fixed_list[i].toFixed(0),
      "high": data.g_upper_list[i].toFixed(3),
      "low": data.g_lower_list[i].toFixed(3)
    })
  }

  fixedModelChartData.push({
    "category": "g(ave) fixed model",
    "measure": data.ave_g,
    "bullet": "diamond",
    "bulletSize": 25,
    "high": data.upper_g_ave,
    "low": data.lower_g_ave
  })

  AmCharts.makeChart(containerId, {
    "type": "serial",
    "theme": "light",
    "rotate": true,
    "fontSize": 18,
    "categoryAxis": {
      "gridAlpha": 0,
    },
    "valueAxes": [{
      "guides": [{
        "value": 2.2,
        "dashLength": 3,
        "lineThickness": 2
      }]
    }],
    "chartCursor": {
      "fullWidth": true,
      "graphBulletSize": 1,
      "categoryBalloonEnabled": false,
      "cursorAlpha": 0.1
    },
    "dataProvider": fixedModelChartData,
    "graphs": [{
      "lineAlpha": 0,
      "bulletField": "bullet",
      "bulletSizeField": "bulletSize",
      "valueField": "measure",
      "lineColor": "#00c20d",
      "balloonText": "<strong>[[value]]</strong> 95%CI([[low]],[[high]])"
    }, {
      "type": "ohlc",
      "highField": "high",
      "lowField": "low",
      "openField": "measure",
      "closeField": "measure",
      "fixedColumnWidth": 1,
      "lineColor": "#00c20d",
      "showBalloon": false
    }],
    "categoryField": "category"
  });
}

function renderRandomModelChart (containerId, data) {
  let randomModelChartData = []

  randomModelChartData.push({
    "category": "",
    "measure": 0,
    "bullet": "",
    "bulletSize": 0,
    "high": 0,
    "low": 0
  })

  for (var i = 0; i < data.g_list.length; i++) {
    randomModelChartData.push({
      "category": data.study_list[i],
      "measure": data.g_list[i].toFixed(3),
      "bullet": "square",
      "bulletSize": data.weight_random_list[i].toFixed(0),
      "high": data.g_upper_list[i].toFixed(3),
      "low": data.g_lower_list[i].toFixed(3)
    })
  }

  randomModelChartData.push({
    "category": "g(ave) random model",
    "measure": data.ave_g,
    "bullet": "diamond",
    "bulletSize": 25,
    "high": data.upper_g_ave,
    "low": data.lower_g_ave
  })

  AmCharts.makeChart(containerId, {
    "type": "serial",
    "theme": "light",
    "rotate": true,
    "fontSize": 18,
    "categoryAxis": {
      "gridAlpha": 0,
    },
    "valueAxes": [{
      "guides": [{
        "value": 2.2,
        "dashLength": 3,
        "lineThickness": 2
      }]
    }],
    "chartCursor": {
      "fullWidth": true,
      "graphBulletSize": 1,
      "categoryBalloonEnabled": false,
      "cursorAlpha": 0.1
    },
    "dataProvider": randomModelChartData,
    "graphs": [{
      "lineAlpha": 0,
      "bulletField": "bullet",
      "bulletSizeField": "bulletSize",
      "valueField": "measure",
      "lineColor": "#00c20d",
      "balloonText": "<strong>[[value]]</strong> 95%CI([[low]],[[high]])"
    }, {
      "type": "ohlc",
      "highField": "high",
      "lowField": "low",
      "openField": "measure",
      "closeField": "measure",
      "fixedColumnWidth": 1,
      "lineColor": "#00c20d",
      "showBalloon": false
    }],
    "categoryField": "category"
  });
}
