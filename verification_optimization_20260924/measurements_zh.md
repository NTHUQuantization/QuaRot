# Verification 最佳化實測表

此表僅由已存在的量測 JSON 產生；未完成的 stage 不填入預估收益。
Target latency 以 HIP event 的 median 表示；kernel sum 不等於端到端 latency。
PARD-2 小樣本 smoke 保留原 checkpoint、greedy acceptance 與資料集 hash 檢查；不是正式三資料集 qualification。

## Target-only，相同 context=128

|Stage|M|GPU kernel launches|CPU launch APIs|Event median ms|Wall median ms|Strict checks|
|---|---:|---:|---:|---:|---:|---|
|baseline|1|2136|2136|30.937|30.983|PASS|
|baseline|15|6276|6276|80.110|80.151|PASS|
|baseline|16|6564|6564|80.909|80.941|PASS|
|had_only|1|1848|1848|27.401|27.438|PASS|
|had_only|15|1956|1956|35.445|35.484|PASS|
|had_only|16|1956|1956|34.855|34.889|PASS|
|norm_only|1|1704|1704|27.075|27.108|PASS|
|norm_only|15|5844|5844|75.387|75.428|PASS|
|norm_only|16|6132|6132|78.793|78.838|PASS|
|metadata_only|1|2138|2138|32.386|32.420|PASS|
|metadata_only|15|6278|6278|74.621|74.658|PASS|
|metadata_only|16|6566|6566|78.842|78.881|PASS|
|graph_with_metadata|1|2138|2138|32.277|32.311|PASS|
|graph_with_metadata|15|6278|2|55.929|55.962|PASS|
|graph_with_metadata|16|6566|2|57.604|57.664|PASS|
|chunk_only|1|912|912|19.571|19.606|PASS|
|chunk_only|15|984|984|27.780|27.813|PASS|
|chunk_only|16|984|984|28.194|28.232|PASS|
|had_norm|1|1416|1416|22.991|23.030|PASS|
|had_norm|15|1524|1524|30.499|30.538|PASS|
|had_norm|16|1524|1524|30.987|31.023|PASS|
|had_norm_metadata|1|1418|1418|22.874|22.906|PASS|
|had_norm_metadata|15|1526|1526|23.831|23.864|PASS|
|had_norm_metadata|16|1526|1526|24.600|24.638|PASS|
|had_norm_metadata_graph|1|1418|1418|22.407|22.444|PASS|
|had_norm_metadata_graph|15|1526|2|22.695|22.737|PASS|
|had_norm_metadata_graph|16|1526|2|22.689|22.720|PASS|
|all|1|482|482|17.955|17.994|PASS|
|all|15|554|2|18.975|19.016|PASS|
|all|16|554|2|19.041|19.084|PASS|
|baseline_repeat|1|2136|2136|32.968|33.012|PASS|
|baseline_repeat|15|6276|6276|80.401|80.444|PASS|
|baseline_repeat|16|6564|6564|82.617|82.654|PASS|

## PARD-2 實際生成

|Stage|Mode|Steady tokens/s|E2E tokens/s|Verify ms/step|Mean accept|Output parity|Reject steps|
|---|---|---:|---:|---:|---:|---|---:|
|baseline|pard2-td|30.359|25.838|81.614|3.600|True|20|
|baseline|pard2-ti|22.358|19.515|83.109|2.769|True|26|
|had_only|pard2-td|52.217|42.152|34.601|3.600|True|20|
|had_only|pard2-ti|38.897|33.203|34.615|2.769|True|26|
|norm_only|pard2-td|32.433|27.729|74.582|3.600|True|20|
|norm_only|pard2-ti|23.393|20.966|78.395|2.769|True|26|
|metadata_only|pard2-td|32.326|27.251|75.104|3.600|True|20|
|metadata_only|pard2-ti|24.277|21.459|74.887|2.769|True|26|
|graph_with_metadata|pard2-td|38.809|32.466|57.562|3.600|True|20|
|graph_with_metadata|pard2-ti|29.261|25.708|57.186|2.769|True|26|
|chunk_only|pard2-td|57.302|43.781|28.726|3.600|True|20|
|chunk_only|pard2-ti|43.217|36.582|27.646|2.769|True|26|
|had_norm|pard2-td|56.377|45.295|29.479|3.600|True|20|
|had_norm|pard2-ti|42.273|35.978|29.615|2.769|True|26|
|had_norm_metadata|pard2-td|62.388|49.323|23.759|3.600|True|20|
|had_norm_metadata|pard2-ti|46.973|38.217|23.800|2.769|True|26|
|had_norm_metadata_graph|pard2-td|62.936|49.726|22.821|3.600|True|20|
|had_norm_metadata_graph|pard2-ti|48.227|40.405|22.635|2.769|True|26|
|all|pard2-td|68.391|53.365|19.146|3.600|True|20|
|all|pard2-ti|51.976|42.726|18.995|2.769|True|26|
|baseline_repeat|pard2-td|30.418|26.031|81.432|3.600|True|20|
|baseline_repeat|pard2-ti|22.849|20.442|81.254|2.769|True|26|

Acceptance 與每輪 accept length 另記錄在 summary.json；輸出 token 相同不代表 candidate 相同。
Eager kernel 數取自 GPU trace；ROCtracer 無法完整列出 graph 內部事件時，以 HIP graph kernel node 枚舉加上 graph 外的 kernel launch API 計數。詳細來源保存在 summary.json；單次 graph API 呼叫不代表單一 GPU kernel。
