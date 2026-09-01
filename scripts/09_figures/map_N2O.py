# -*- coding: utf-8 -*-
# @Time    : 2025/12/30
# @Author  : 55050
# @File    : map_N2O.py
# @Software: PyCharm


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
# ★★★ 关键：引入 JsCode 用于控制前端显示逻辑 ★★★
from pyecharts.commons.utils import JsCode
import warnings

warnings.filterwarnings('ignore')

# 设置字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============ 配置区 ============
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / 'data' / 'figure_inputs' / 'N2O_map_values.xlsx'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'figures' / 'province_maps'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_MAP = OUTPUT_DIR / 'N2O_map.html'
OUTPUT_PNG = OUTPUT_DIR / 'N2O_ranking.png'
OUTPUT_PDF = OUTPUT_DIR / 'N2O_ranking.pdf'
# ================================

print("=" * 60)
print("生成 Nature 风格可视化 (解决文字重叠版)")
print("=" * 60)

# 1. 读取数据
try:
    df = pd.read_excel(INPUT_FILE)
except FileNotFoundError:
    print(f"❌ 错误：找不到文件 {INPUT_FILE}")
    exit()

province_col = df.columns[1]
n2o_col = 'N2O排放速率'

# --- 核心映射表 ---
standard_names = {
    '北京': '北京市', '天津': '天津市', '上海': '上海市', '重庆': '重庆市',
    '河北': '河北省', '山西': '山西省', '辽宁': '辽宁省', '吉林': '吉林省',
    '黑龙江': '黑龙江省', '江苏': '江苏省', '浙江': '浙江省', '安徽': '安徽省',
    '福建': '福建省', '江西': '江西省', '山东': '山东省', '河南': '河南省',
    '湖北': '湖北省', '湖南': '湖南省', '广东': '广东省', '海南': '海南省',
    '四川': '四川省', '贵州': '贵州省', '云南': '云南省', '陕西': '陕西省',
    '甘肃': '甘肃省', '青海': '青海省', '台湾': '台湾省',
    '内蒙古': '内蒙古自治区', '广西': '广西壮族自治区', '西藏': '西藏自治区',
    '宁夏': '宁夏回族自治区', '新疆': '新疆维吾尔自治区',
    '香港': '香港特别行政区', '澳门': '澳门特别行政区'
}

name_map_dict = {
    '北京市': 'Beijing', '天津市': 'Tianjin', '河北省': 'Hebei', '山西省': 'Shanxi',
    '内蒙古自治区': 'Inner Mongolia', '辽宁省': 'Liaoning', '吉林省': 'Jilin',
    '黑龙江省': 'Heilongjiang', '上海市': 'Shanghai', '江苏省': 'Jiangsu',
    '浙江省': 'Zhejiang', '安徽省': 'Anhui', '福建省': 'Fujian', '江西省': 'Jiangxi',
    '山东省': 'Shandong', '河南省': 'Henan', '湖北省': 'Hubei', '湖南省': 'Hunan',
    '广东省': 'Guangdong', '广西壮族自治区': 'Guangxi', '海南省': 'Hainan',
    '重庆市': 'Chongqing', '四川省': 'Sichuan', '贵州省': 'Guizhou',
    '云南省': 'Yunnan', '西藏自治区': 'Tibet', '陕西省': 'Shaanxi', '甘肃省': 'Gansu',
    '青海省': 'Qinghai', '宁夏回族自治区': 'Ningxia', '新疆维吾尔自治区': 'Xinjiang',
    '香港特别行政区': 'Hong Kong', '澳门特别行政区': 'Macau', '台湾省': 'Taiwan'
}

# 2. 数据准备
clean_provinces = []
english_labels = []
for p in df[province_col]:
    p_str = str(p).strip()
    full_cn_name = standard_names.get(p_str, p_str if p_str in standard_names.values() else p_str)
    clean_provinces.append(full_cn_name)
    english_labels.append(name_map_dict.get(full_cn_name, full_cn_name))

df['Eng_Province'] = english_labels

# ★ 配色方案 (Nature Red)
PREMIUM_COLORS = [
    "#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59",
    "#ef6548", "#d7301f", "#b30000", "#7f0000"
]

# 3. 生成地图 (HTML)
print("\n[3/4] 生成地图 (已优化重叠问题)...")
try:
    from pyecharts import options as opts
    from pyecharts.charts import Map
    from pyecharts.globals import ThemeType

    map_data = list(zip(df['Eng_Province'], df[n2o_col]))

    # ★★★ 关键：定义 JS 函数来过滤特定名字 ★★★
    # 逻辑：如果名字是 Hong Kong 或 Macau，返回空字符串；否则返回名字
    custom_label_js = JsCode("""
        function(params) {
            if (params.name === 'Hong Kong' || params.name === 'Macau') {
                return ''; 
            }
            return params.name;
        }
    """)

    map_chart = (
        Map(init_opts=opts.InitOpts(width="1400px", height="900px", theme=ThemeType.LIGHT,
                                    page_title="Premium N2O Map"))
        .add(
            series_name="N2O Rate",
            data_pair=map_data,
            maptype="china",
            is_map_symbol_show=False,
            name_map=name_map_dict,

            # ★★★ 在这里应用 JS 过滤器 ★★★
            label_opts=opts.LabelOpts(
                is_show=True,
                formatter=custom_label_js,  # 使用我们写的 JS 函数
                font_size=11,
                color="#444444",
                font_family="Arial"
            ),

            itemstyle_opts=opts.ItemStyleOpts(border_color="#fff", border_width=1)
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="N2O Emission Intensity",
                subtitle=f"Unit: Emission Rate",
                pos_left="5%", pos_top="5%",
                title_textstyle_opts=opts.TextStyleOpts(font_family="Arial", font_size=26, color="#000"),
            ),
            visualmap_opts=opts.VisualMapOpts(
                min_=df[n2o_col].min(), max_=df[n2o_col].max(),
                range_text=["High", "Low"], is_piecewise=False,
                pos_right="5%", pos_bottom="10%", orient="vertical",
                range_color=PREMIUM_COLORS,
                textstyle_opts=opts.TextStyleOpts(color="#555", font_family="Arial"),
                item_width=15, item_height=180
            ),
            tooltip_opts=opts.TooltipOpts(formatter="{b}<br/>Rate: {c:.4f}")
        )
    )
    map_chart.render(OUTPUT_MAP)
    print(f"✓ 地图 HTML 已保存: {OUTPUT_MAP}")

except Exception as e:
    print(f"⚠ 地图错误: {e}")

# 4. 生成条形图 (保持不变)
print("\n[4/4] 生成条形图 (PDF & PNG)...")
try:
    fig, ax = plt.subplots(figsize=(10, 12))
    df_sorted = df.sort_values(n2o_col, ascending=True)

    premium_cmap = LinearSegmentedColormap.from_list("premium_red", PREMIUM_COLORS)
    norm = plt.Normalize(vmin=df_sorted[n2o_col].min(), vmax=df_sorted[n2o_col].max())

    bars = ax.barh(range(len(df_sorted)), df_sorted[n2o_col],
                   color=premium_cmap(norm(df_sorted[n2o_col])), height=0.65)

    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['Eng_Province'], fontsize=11, fontfamily='Arial', color='#333')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#ccc')
    ax.spines['bottom'].set_linewidth(0.8)

    ax.tick_params(axis='x', colors='#666')
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', alpha=0.3, linestyle=':', color='gray')

    ax.set_xlabel('Emission Rate', fontsize=12, fontfamily='Arial', fontweight='bold', color='#444', labelpad=15)
    ax.set_title('Provincial N2O Emissions Ranking', fontsize=16, fontfamily='Arial', fontweight='bold', color='#222',
                 pad=25, loc='left')

    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        ax.text(row[n2o_col] + (df[n2o_col].max() * 0.02), i, f'{row[n2o_col]:.3f}',
                va='center', fontsize=9, fontfamily='Arial', color='#666')

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(OUTPUT_PDF, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"✓ 条形图已保存: {OUTPUT_PNG} 和 {OUTPUT_PDF}")

except Exception as e:
    import traceback

    traceback.print_exc()
    print(f"✗ 条形图错误: {e}")

print("\nDone.")