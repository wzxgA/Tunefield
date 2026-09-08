import { useEffect, useRef } from "react";
import * as echarts from "echarts/core";
import { LineChart } from "echarts/charts";
import { GridComponent, TooltipComponent } from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

// 按需注册（贴合轻量约束，不引入全量 echarts）
echarts.use([LineChart, GridComponent, TooltipComponent, CanvasRenderer]);

// Loss 折线：points = [{step, value}]
export default function LossChart({ points }) {
  const elRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    chartRef.current = echarts.init(elRef.current);
    const onResize = () => chartRef.current && chartRef.current.resize();
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      if (chartRef.current) chartRef.current.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    chart.setOption({
      grid: { left: 44, right: 14, top: 18, bottom: 26 },
      xAxis: {
        type: "category",
        data: points.map((p) => p.step),
        axisLine: { lineStyle: { color: "#d8d0bf" } },
        axisLabel: { color: "#8b8477", fontSize: 11 },
      },
      yAxis: {
        type: "value",
        scale: true,
        splitLine: { lineStyle: { color: "#efe9dc" } },
        axisLabel: { color: "#8b8477", fontSize: 11 },
      },
      series: [
        {
          type: "line",
          data: points.map((p) => p.value),
          smooth: true,
          symbolSize: 5,
          lineStyle: { width: 2, color: "#b0955d" },
          itemStyle: { color: "#b0955d" },
          areaStyle: { color: "rgba(176,149,93,0.10)" },
        },
      ],
      tooltip: { trigger: "axis" },
    });
  }, [points]);

  return <div ref={elRef} className="loss-chart" />;
}
