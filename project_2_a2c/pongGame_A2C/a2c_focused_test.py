import sys

sys.path.append("../src/")
import wandb
import os
import time
import matplotlib.pyplot as plt
from config import *
from pong_wrapper import *
from process_image import *
from utilities import *
from network import *
from agent import *

# A2C 집중 최적화 조합들 (8개)
A2C_FOCUSED_COMBINATIONS = [
    # Phase 1: 기본 A2C 최적화 (4개)
    {
        "name": "a2c_baseline_optimized",
        "agent": "A2C",
        "lr": 7e-4,
        "value_c": 1.0,
        "entropy_c": 0.015,
        "batch_size": 128,
        "n_updates": 3000
    },
    {
        "name": "a2c_higher_lr",
        "agent": "A2C",
        "lr": 1.5e-3,
        "value_c": 1.5,
        "entropy_c": 0.02,
        "batch_size": 128,
        "n_updates": 3000
    },
    {
        "name": "a2c_bigger_batch",
        "agent": "A2C",
        "lr": 1e-3,
        "value_c": 2.0,
        "entropy_c": 0.015,
        "batch_size": 256,
        "n_updates": 3000
    },
    {
        "name": "a2c_high_exploration",
        "agent": "A2C",
        "lr": 8e-4,
        "value_c": 1.0,
        "entropy_c": 0.03,
        "batch_size": 128,
        "n_updates": 3000
    },

    # Phase 2: 더 aggressive한 A2C 설정 (4개)
    {
        "name": "a2c_very_high_lr",
        "agent": "A2C",
        "lr": 2e-3,
        "value_c": 2.0,
        "entropy_c": 0.01,
        "batch_size": 192,
        "n_updates": 4000
    },
    {
        "name": "a2c_balanced_strong",
        "agent": "A2C",
        "lr": 1.2e-3,
        "value_c": 3.0,
        "entropy_c": 0.025,
        "batch_size": 256,
        "n_updates": 4000
    },
    {
        "name": "a2c_conservative_stable",
        "agent": "A2C",
        "lr": 5e-4,
        "value_c": 0.5,
        "entropy_c": 0.01,
        "batch_size": 128,
        "n_updates": 4000
    },
    {
        "name": "a2c_final_candidate",
        "agent": "A2C",
        "lr": 1e-3,
        "value_c": 1.5,
        "entropy_c": 0.018,
        "batch_size": 192,
        "n_updates": 5000  # 가장 긴 학습
    }
]


def run_a2c_test(params):
    """A2C 전용 테스트"""
    print(f"\n{'=' * 60}")
    print(f"🔥 A2C 테스트: {params['name']}")
    print(f"LR: {params['lr']}, VALUE_C: {params['value_c']}")
    print(f"ENTROPY_C: {params['entropy_c']}, BATCH: {params['batch_size']}")
    print(f"UPDATES: {params['n_updates']}")
    print(f"{'=' * 60}")

    try:
        print("1. A2C 설정 준비 중...")

        # A2C 전용 config
        test_config = {
            "agent": "A2C",
            "input_shape": INPUT_SHAPE,
            "batch_size": params['batch_size'],
            "n_updates": params['n_updates'],
            "hidden": HIDDEN,
            "learning_rate": params['lr'],
            "gamma": GAMMA,
            "value_c": params['value_c'],
            "entropy_c": params['entropy_c'],
            "std_adv": STD_ADV,
            "test_name": params['name'],
            "operating_system": os.name
        }

        print("2. WandB 초기화 중...")
        run_name = f"A2C_{params['name']}_lr{params['lr']}_vc{params['value_c']}_ent{params['entropy_c']}"

        wandb.init(
            project="tensorflow2_pong_a2c_focused",
            name=run_name,
            tags=["A2C", "CNN", "RL", "atari_pong", "focused_test", params['name']],
            config=test_config,
            mode="online"
        )

        print(f"WandB URL: {wandb.run.url}")

        print("3. 환경 및 모델 초기화 중...")
        pw = PongWrapper(ENV_NAME, history_length=4)
        model = Model(num_actions=pw.env.action_space.n, hidden=HIDDEN)

        print("4. A2C 에이전트 초기화 중...")
        agent = Agent(
            model=model,
            save_path=f"../model/a2c_focused_{params['name']}",
            load_path=None,
            lr=params['lr'],
            gamma=GAMMA,
            value_c=params['value_c'],
            entropy_c=params['entropy_c'],
            clip_ratio=0.2,  # A2C에서는 사용 안되지만 기본값
            std_adv=STD_ADV,
            agent="A2C",
            input_shape=INPUT_SHAPE,
            batch_size=params['batch_size'],
            updates=params['n_updates']
        )

        print("5. A2C 학습 시작...")
        start_time = time.time()
        rewards_history = agent.train(pw)
        end_time = time.time()
        print("A2C 학습 완료!")

        # 상세 분석
        total_time = end_time - start_time

        # 단계별 성능 분석
        episodes_count = len(rewards_history)
        if episodes_count >= 200:
            first_50 = np.mean(rewards_history[:50])
            second_50 = np.mean(rewards_history[50:100]) if episodes_count >= 100 else first_50
            third_50 = np.mean(rewards_history[100:150]) if episodes_count >= 150 else second_50
            last_50 = np.mean(rewards_history[-50:])
        else:
            quarter = episodes_count // 4
            first_50 = np.mean(rewards_history[:quarter]) if quarter > 0 else -21
            second_50 = np.mean(rewards_history[quarter:quarter * 2]) if quarter > 0 else first_50
            third_50 = np.mean(rewards_history[quarter * 2:quarter * 3]) if quarter > 0 else second_50
            last_50 = np.mean(rewards_history[quarter * 3:]) if quarter > 0 else third_50

        max_reward = max(rewards_history) if rewards_history else -21
        min_reward = min(rewards_history) if rewards_history else -21
        overall_improvement = last_50 - first_50

        # 학습 안정성 점수
        if episodes_count >= 100:
            recent_std = np.std(rewards_history[-100:])
            stability_score = max(0, 5 - recent_std)  # 표준편차가 낮을수록 높은 점수
        else:
            stability_score = 0

        # 종합 점수 계산
        final_score = (
                (max_reward + 21) * 2 +  # 최고 성능 (0~42점)
                (overall_improvement + 5) * 3 +  # 개선도 (-5~+5 → 0~30점)
                stability_score * 2  # 안정성 (0~10점)
        )

        print(f"\n📊 A2C 성능 분석:")
        print(f"    첫 50 에피소드 평균: {first_50:.2f}")
        print(f"    마지막 50 에피소드 평균: {last_50:.2f}")
        print(f"    전체 개선도: {overall_improvement:.2f}")
        print(f"    최고 리워드: {max_reward:.2f}")
        print(f"    안정성 점수: {stability_score:.2f}")
        print(f"    종합 점수: {final_score:.2f}")

        # 결과 저장
        timestamp = time.strftime('%Y%m%d%H%M')
        save_dir = f"../model/a2c_focused_{params['name']}/save_agent_{timestamp}"
        plot_dir = os.path.join(save_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        # 상세 그래프 생성
        plt.figure(figsize=(15, 10))

        # 전체 리워드 그래프
        plt.subplot(2, 3, 1)
        plt.plot(rewards_history, alpha=0.7)
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title(f"Episode Rewards - {params['name']}")
        plt.grid(True)

        # Moving average 그래프
        plt.subplot(2, 3, 2)
        if len(rewards_history) > 20:
            window = min(50, len(rewards_history) // 4)
            moving_avg = []
            for i in range(window, len(rewards_history)):
                moving_avg.append(np.mean(rewards_history[i - window:i]))
            plt.plot(moving_avg)
            plt.xlabel("Episodes")
            plt.ylabel(f"{window}-Episode Moving Average")
            plt.title("Moving Average Rewards")
            plt.grid(True)

        # 단계별 성능 비교
        plt.subplot(2, 3, 3)
        stages = ['First 50', 'Second 50', 'Third 50', 'Last 50']
        values = [first_50, second_50, third_50, last_50]
        plt.bar(stages, values, color=['red', 'orange', 'yellow', 'green'])
        plt.ylabel("Average Reward")
        plt.title("Performance by Stage")
        plt.xticks(rotation=45)

        # 최근 100 에피소드 분포
        plt.subplot(2, 3, 4)
        recent_rewards = rewards_history[-100:] if len(rewards_history) >= 100 else rewards_history
        plt.hist(recent_rewards, bins=20, alpha=0.7)
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.title("Recent Reward Distribution")

        # 개선 추세
        plt.subplot(2, 3, 5)
        if len(rewards_history) > 10:
            chunk_size = max(10, len(rewards_history) // 20)
            chunk_means = []
            for i in range(0, len(rewards_history), chunk_size):
                chunk = rewards_history[i:i + chunk_size]
                chunk_means.append(np.mean(chunk))
            plt.plot(chunk_means, marker='o')
            plt.xlabel(f"Chunks of {chunk_size} episodes")
            plt.ylabel("Average Reward")
            plt.title("Learning Progress")
            plt.grid(True)

        # 종합 점수 표시
        plt.subplot(2, 3, 6)
        score_components = ['Max Reward', 'Improvement', 'Stability', 'Total']
        score_values = [
            (max_reward + 21) * 2,
            (overall_improvement + 5) * 3,
            stability_score * 2,
            final_score
        ]
        colors = ['blue', 'green', 'orange', 'red']
        plt.bar(score_components, score_values, color=colors)
        plt.ylabel("Score")
        plt.title("Performance Score Breakdown")
        plt.xticks(rotation=45)

        plt.tight_layout()
        plot_path = os.path.join(plot_dir, "a2c_detailed_analysis.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        # 모델 저장 및 테스트
        model_path = f"{save_dir}/model.tf"
        video_path = f"{save_dir}/test_play.mp4"

        agent.save_model(model_path)
        final_test_reward = agent.test(pw, render=False, record_path=video_path)

        # WandB에 상세 결과 로그
        wandb.log({
            "final_test_reward": final_test_reward,
            "max_reward_achieved": max_reward,
            "min_reward_achieved": min_reward,
            "overall_improvement": overall_improvement,
            "first_50_avg": first_50,
            "last_50_avg": last_50,
            "stability_score": stability_score,
            "final_score": final_score,
            "total_training_time_minutes": total_time / 60,
            "total_episodes": episodes_count
        })

        print(f"[✔] {params['name']} 완료! 종합 점수: {final_score:.2f}")
        print(f"    모델: {model_path}")
        print(f"    영상: {video_path}")

        return {
            "name": params['name'],
            "final_score": final_score,
            "max_reward": max_reward,
            "improvement": overall_improvement,
            "stability": stability_score,
            "final_test_reward": final_test_reward,
            "time_minutes": total_time / 60,
            "episodes": episodes_count,
            "params": params
        }

    except Exception as e:
        print(f"[✖] {params['name']} 오류: {e}")
        return None

    finally:
        wandb.finish()


def main():
    """A2C 집중 테스트 메인"""
    print("🎯 A2C 집중 최적화 테스트 시작!")
    print(f"총 {len(A2C_FOCUSED_COMBINATIONS)}개 A2C 조합 테스트")
    print("PPO는 포기하고 A2C로 돌파구를 찾아보자! 💪")

    results = []
    start_total_time = time.time()

    for i, params in enumerate(A2C_FOCUSED_COMBINATIONS, 1):
        print(f"\n🔄 A2C 테스트 진행률: {i}/{len(A2C_FOCUSED_COMBINATIONS)}")
        estimated_time = (len(A2C_FOCUSED_COMBINATIONS) - i) * 35  # A2C는 약 35분 추정
        print(f"⏰ 예상 남은 시간: {estimated_time // 60}시간 {estimated_time % 60}분")

        result = run_a2c_test(params)
        if result:
            results.append(result)

        # 메모리 정리
        if i < len(A2C_FOCUSED_COMBINATIONS):
            print("다음 A2C 테스트 준비 중... (10초 대기)")
            time.sleep(10)

    # 최종 A2C 결과 분석
    total_time = time.time() - start_total_time
    print(f"\n{'=' * 80}")
    print("🏆 A2C 집중 테스트 완료!")
    print(f"총 소요 시간: {total_time / 3600:.1f}시간")
    print(f"\n📊 A2C 성능 순위 (종합 점수 기준):")
    print("-" * 80)

    if results:
        results.sort(key=lambda x: x['final_score'], reverse=True)

        print(f"{'순위':<4} {'이름':<25} {'종합점수':<8} {'최고리워드':<8} {'개선도':<8} {'안정성':<8}")
        print("-" * 80)

        for i, result in enumerate(results, 1):
            print(
                f"{i:<4} {result['name']:<25} {result['final_score']:<8.1f} {result['max_reward']:<8.1f} {result['improvement']:<8.2f} {result['stability']:<8.1f}")

        print(f"\n🥇 최고 A2C 조합:")
        best = results[0]
        print(f"    이름: {best['name']}")
        print(f"    종합 점수: {best['final_score']:.1f}")
        print(f"    최고 리워드: {best['max_reward']:.1f}")
        print(f"    개선도: {best['improvement']:.2f}")
        print(f"\n🔧 최적 A2C 파라미터:")
        for key, value in best['params'].items():
            if key != 'name':
                print(f"    {key}: {value}")

        print(f"\n🚀 다음 단계:")
        print(f"    1. 최고 성능 A2C 조합으로 10000+ 업데이트 장기 학습")
        print(f"    2. 상위 3개 조합 세부 튜닝")
        print(f"    3. 네트워크 구조 개선 시도")

    else:
        print("❌ 성공한 A2C 테스트가 없습니다.")

    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()