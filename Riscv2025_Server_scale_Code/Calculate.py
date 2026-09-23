def calculate_diet_recommendations(nutrition_total):
    """
    根據每日建議攝取量計算飲食建議。
    
    參數:
        nutrition_total: 包含已檢測食物總營養值的字典
        
    回傳:
        包含剩餘建議攝取量和達成百分比的字典
    """
    # 標準每日建議攝取量（基於一般成人）
    daily_recommended = {
        'calories': 2000,      # 卡路里
        'protein': 60,         # 克
        'carbs': 300,          # 克
        'fiber': 25,           # 克
        'fat': 65              # 克（舊版FDA每日參考值）
    }
    
    recommendations = {}
    
    # 計算剩餘營養素和達成百分比
    for nutrient, recommended in daily_recommended.items():
        consumed = nutrition_total.get(nutrient, 0)
        remaining = max(0, recommended - consumed)
        percentage = min(100, round((consumed / recommended) * 100, 1)) if recommended > 0 else 0
        
        recommendations[nutrient] = {
            'consumed': consumed,
            'recommended': recommended,
            'remaining': remaining,
            'percentage': percentage
        }
    
    return recommendations