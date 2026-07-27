import logging
from django.shortcuts import render
from django.http import JsonResponse
from .models import PredictionHistory

logger = logging.getLogger(__name__)


SUPPORTED_CLUBS = [
    'Arsenal',
    'Chelsea',
    'Liverpool',
    'Manchester City',
    'Manchester United',
]

ALL_TEAMS = [
    'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Brighton',
    'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham',
    'Ipswich', 'Leeds', 'Leicester', 'Liverpool', 'Luton',
    'Manchester City', 'Manchester United', 'Newcastle', 'Norwich',
    'Nottingham Forest', 'Sheffield United', 'Southampton', 'Tottenham',
    'Watford', 'West Brom', 'West Ham', 'Wolverhampton',
]


def predict(request):
    context = {
        'clubs': ALL_TEAMS,
        'supported_clubs': SUPPORTED_CLUBS,
    }
    return render(request, 'prediction/prediction.html', context)


def history(request):
    predictions = PredictionHistory.objects.all()[:50]
    context = {
        'predictions': predictions,
    }
    return render(request, 'prediction/history.html', context)


def api_predict(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    home_team = request.POST.get('home_team', '')
    away_team = request.POST.get('away_team', '')

    if not home_team or not away_team:
        return JsonResponse({'error': 'Both teams required'}, status=400)

    if home_team == away_team:
        return JsonResponse({'error': 'Teams must be different'}, status=400)

    if home_team not in ALL_TEAMS or away_team not in ALL_TEAMS:
        return JsonResponse({'error': 'Invalid team selection'}, status=400)

    if home_team not in SUPPORTED_CLUBS and away_team not in SUPPORTED_CLUBS:
        return JsonResponse({'error': 'At least one team must be a supported club (Arsenal, Liverpool, Manchester City, Chelsea, Manchester United)'}, status=400)

    try:
        from prediction.ml_service import PredictionService
        service = PredictionService()
        result = service.predict(home_team, away_team)
    except Exception as e:
        logger.exception('Prediction failed')
        return JsonResponse({
            'predicted_result': 'H',
            'predicted_label': 'Home Win',
            'confidence': 45.0,
            'probabilities': {'home': 45.0, 'draw': 25.0, 'away': 30.0},
            'shap_features': [],
            'error': str(e),
        })

    try:
        PredictionHistory.objects.create(
            home_team=home_team,
            away_team=away_team,
            predicted_result=result['predicted_result'],
            predicted_label=result['predicted_label'],
            confidence=result['confidence'],
            prob_home=result['probabilities']['home'],
            prob_draw=result['probabilities']['draw'],
            prob_away=result['probabilities']['away'],
            shap_features=result.get('shap_features', []),
        )
    except Exception as e:
        logger.warning('Failed to save prediction history: %s', e)

    return JsonResponse(result)
