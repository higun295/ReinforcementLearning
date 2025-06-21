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

# 5-6시간 확장 테스트용 조합들 (총 12개)
EXTENDED_TEST_COMBINATIONS = [
    # Phase 1: Value Function 문제 해결 집중 (4개)
    {
        "name": "emergency_high_lr",
        "lr": 2e-3,
        "value_c": 3.0,
        "clip_ratio": 0.3,
        "entropy_c": 0.01,
        "batch_size": 128,
        "agent": "PPO"
    },
    {
        "name": "extreme_value_focus",
        "lr": 1e-3,
        "value_c": 10.0,
        "clip_ratio": 0.2,
        "entropy_c": 0.005,
        "batch_size": 256,
        "agent": "PPO"
    },
    {
        "name": "try_a2c_high_lr",
        "lr": 1e-3,
        "value_c": 2.0,
        "entropy_c": 0.01,
        "batch_size": 128,
        "agent": "A2C"
    },
    {
        "name": "try_a2c_baseline",
        "lr": 7e-4,
        "value_c": 1.0,
        "entropy_c": 0.015,
        "batch_size": 128,
        "agent": "A2C"
    },

    # Phase 2: 극단적 파라미터 실험 (4개)
    {
        "name": "very_high_lr",
        "lr": 5e-3,
        "value_c": 2.0,
        "clip_ratio": 0.4,
        "entropy_c": 0.005,
        "batch_size": 64,
        "agent": "PPO"
    },
    {
        "name": "huge_batch",
        "lr": 1e-3,
        "value_c": 2.0,
        "clip_ratio": 0.2,
        "entropy_c": 0.01,
        "batch_size": 512,
        "agent": "PPO"
    },
    {
        "name": "high_exploration_fixed",
        "lr": 1e-3,
        "value_c": 3.0,
        "clip_ratio": 0.2,
        "entropy_c": 0.05,
        "batch_size": 128,
        "agent": "PPO"
    },
    {
        "name": "conservative_but_stable",
        "lr": 5e-4,
        "value_c": 5.0,
        "clip_ratio": 0.1,
        "entropy_c": 0.01,
        "batch_size": 256,
        "agent": "PPO"
    },

    # Phase 3: 유망한 조합 longer run (4개 - 더 긴 학습)
    {
        "name": "promising_long_1",
        "lr": 1e-3,
        "value_c": 3.0,
        "clip_ratio": 0.2,
        "entropy_c": 0.01,
        "batch_size": 128,
        "agent": "PPO",
        "n_updates": 3000  # 더 긴 학습
    },
    {
        "name": "promising_long_2",
        "lr": 2e-3,
        "value_c": 2.0,
        "clip_ratio": 0.3,
        "entropy_c": 0.005,
        "batch_size": 256,
        "agent": "PPO",
        "n_updates": 3000
    },
    {
        "name": "a2c_long_test",
        "lr": 1e-3,
        "value_c": 2.0,
        "entropy_c": 0.01,
        "batch_size": 128,
        "agent": "A2C",
        "n_updates": 3000
    },
    {
        "name": "final_best_guess",
        "lr": 1.5e-3,
        "value_c": 4.0,
        "clip_ratio": 0.25,
        "entropy_c": 0.008,
        "batch_size": 192,
        "agent": "PPO",
        "n_updates": 3000
    }
]

# 기본 테스트 설정
DEFAULT_BATCH_SIZE = 128
DEFAULT_N_UPDATES = 2000


def run_single_test(params):
    """단일 파라미터 조합 테스트"""
    print(f"\n{'=' * 60}")
    print(f"테스트 시작: {params['name']}")
    print(f"Agent: {params.get('agent', 'PPO')}")
    print(f"LR: {params['lr']}, VALUE_C: {params.get('value_c', 0.5)}")
    print(f"CLIP: {params.get('clip_ratio', 0.2)}, ENTROPY: {params['entropy_c']}")
    print(f"BATCH: {params.get('batch_size', DEFAULT_BATCH_SIZE)}")
    print(f"UPDATES: {params.get('n_updates', DEFAULT_N_UPDATES)}")
    print(f"{'=' * 60}")

    try:
        print("1. 설정 준비 중...")

        # 파라미터 기본값 설정
        test_lr = params['lr']
        test_value_c = params.get('value_c', VALUE_C)
        test_clip_ratio = params.get('clip_ratio', CLIP_RATIO)
        test_entropy_c = params['entropy_c']
        test_batch_size = params.get('batch_size', DEFAULT_BATCH_SIZE)
        test_n_updates = params.get('n_updates', DEFAULT_N_UPDATES)
        test_agent = params.get('agent', AGENT)

        # 현재 테스트용 config 생성
        test_config = CONFIG_WANDB.copy()
        test_config.update({
            "batch_size": test_batch_size,
            "n_updates": test_n_updates,
            "learning_rate": test_lr,
            "value_c": test_value_c,
            "entropy_c": test_entropy_c,
            "clip_ratio": test_clip_ratio,
            "agent": test_agent,
            "test_name": params['name']
        })

        print("2. WandB 초기화 중...")
        # WandB 초기화
        run_name = f"{test_agent}_{params['name']}_lr{test_lr}_vc{test_value_c}"

        print(f"   프로젝트명: tensorflow2_pong_extended_test")
        print(f"   실행명: {run_name}")

        wandb.init(
            project="tensorflow2_pong_extended_test",
            name=run_name,
            tags=["extended_test", test_agent, "CNN", "RL", "atari_pong", params['name']],
            config=test_config,
            mode="online"
        )

        print(f"WandB URL: {wandb.run.url}")

        print("3. 환경 초기화 중...")
        pw = PongWrapper(ENV_NAME, history_length=4)
        print("4. 모델 초기화 중...")
        model = Model(num_actions=pw.env.action_space.n, hidden=HIDDEN)
        print("5. 에이전트 초기화 중...")

        # 에이전트 초기화
        agent = Agent(
            model=model,
            save_path=f"../model/extended_test_{params['name']}",
            load_path=None,
            lr=test_lr,
            gamma=GAMMA,
            value_c=test_value_c,
            entropy_c=test_entropy_c,
            clip_ratio=test_clip_ratio,
            std_adv=STD_ADV,
            agent=test_agent,
            input_shape=INPUT_SHAPE,
            batch_size=test_batch_size,
            updates=test_n_updates
        )

        print("6. 학습 시작...")
        start_time = time.time()
        rewards_history = agent.train(pw)
        end_time = time.time()
        print("학습 완료!")

        # 결과 저장
        timestamp = time.strftime('%Y%m%d%H%M')
        save_dir = f"../model/extended_test_{params['name']}/save_agent_{timestamp}"
        plot_dir = os.path.join(save_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        # 그래프 저장
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(rewards_history)
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title(f"Episode Rewards - {params['name']}")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        if len(rewards_history) > 50:
            moving_avg = []
            window = 50
            for i in range(window, len(rewards_history)):
                moving_avg.append(np.mean(rewards_history[i - window:i]))
            plt.plot(moving_avg)
            plt.xlabel("Episodes")
            plt.ylabel("50-Episode Moving Average")
            plt.title(f"Moving Average Rewards - {params['name']}")
            plt.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(plot_dir, "detailed_reward_plot.png")
        plt.savefig(plot_path)
        plt.close()

        # 모델 저장 및 테스트
        model_path = f"{save_dir}/model.tf"
        video_path = f"{save_dir}/test_play.mp4"

        agent.save_model(model_path)
        final_reward = agent.test(pw, render=False, record_path=video_path)

        # 결과 분석
        total_time = end_time - start_time
        avg_reward_last_100 = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(
            rewards_history)
        avg_reward_last_50 = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else np.mean(rewards_history)
        max_reward = max(rewards_history) if rewards_history else -21
        min_reward = min(rewards_history) if rewards_history else -21

        # 개선도 계산
        first_100 = np.mean(rewards_history[:100]) if len(rewards_history) >= 100 else np.mean(
            rewards_history[:len(rewards_history) // 2])
        improvement = avg_reward_last_100 - first_100

        # 최종 결과 로그
        wandb.log({
            "final_test_reward": final_reward,
            "avg_reward_last_100_episodes": avg_reward_last_100,
            "avg_reward_last_50_episodes": avg_reward_last_50,
            "max_reward_achieved": max_reward,
            "min_reward_achieved": min_reward,
            "improvement_score": improvement,
            "total_training_time_minutes": total_time / 60,
            "total_episodes": len(rewards_history)
        })

        print(f"\n[✔] {params['name']} 테스트 완료!")
        print(f"    최종 테스트 리워드: {final_reward:.2f}")
        print(f"    최근 100 에피소드 평균: {avg_reward_last_100:.2f}")
        print(f"    최고 리워드: {max_reward:.2f}")
        print(f"    개선도: {improvement:.2f}")
        print(f"    소요 시간: {total_time / 60:.1f}분")

        return {
            "name": params['name'],
            "agent": test_agent,
            "final_reward": final_reward,
            "avg_reward_100": avg_reward_last_100,
            "avg_reward_50": avg_reward_last_50,
            "max_reward": max_reward,
            "improvement": improvement,
            "time_minutes": total_time / 60,
            "episodes": len(rewards_history),
            "params": params
        }

    except Exception as e:
        print(f"[✖] {params['name']} 테스트 중 오류 발생: {e}")
        return None

    finally:
        wandb.finish()


def main():
    """메인 함수 - 확장 테스트 실행"""
    print("🚀 5-6시간 확장 하이퍼파라미터 테스트 시작!")
    print(f"총 {len(EXTENDED_TEST_COMBINATIONS)}개 조합 테스트 예정")
    print(f"예상 소요 시간: 5-6시간")

    results = []
    start_total_time = time.time()

    for i, params in enumerate(EXTENDED_TEST_COMBINATIONS, 1):
        print(f"\n🔄 진행률: {i}/{len(EXTENDED_TEST_COMBINATIONS)} ({i / len(EXTENDED_TEST_COMBINATIONS) * 100:.1f}%)")
        estimated_remaining = (len(EXTENDED_TEST_COMBINATIONS) - i) * 25  # 평균 25분 추정
        print(f"⏰ 예상 남은 시간: {estimated_remaining // 60}시간 {estimated_remaining % 60}분")

        result = run_single_test(params)
        if result:
            results.append(result)

        # 다음 테스트 전 메모리 정리
        if i < len(EXTENDED_TEST_COMBINATIONS):
            print("다음 테스트 준비 중... (10초 대기)")
            time.sleep(10)

    # 전체 결과 요약
    total_time = time.time() - start_total_time
    print(f"\n{'=' * 80}")
    print("🎉 모든 확장 테스트 완료!")
    print(f"총 소요 시간: {total_time / 3600:.1f}시간")
    print(f"\n📊 결과 요약 (개선도 기준 정렬):")
    print("-" * 80)

    if results:
        # 결과 정렬 (개선도 기준)
        results.sort(key=lambda x: x['improvement'], reverse=True)

        print(f"{'순위':<4} {'이름':<20} {'Agent':<5} {'개선도':<8} {'최종리워드':<8} {'최고리워드':<8}")
        print("-" * 80)

        for i, result in enumerate(results[:10], 1):  # 상위 10개만 표시
            print(
                f"{i:<4} {result['name']:<20} {result['agent']:<5} {result['improvement']:<8.2f} {result['final_reward']:<8.2f} {result['max_reward']:<8.2f}")

        print(f"\n🏆 최고 성능 조합:")
        best_config = results[0]
        print(f"    이름: {best_config['name']}")
        print(f"    Agent: {best_config['agent']}")
        print(f"    개선도: {best_config['improvement']:.2f}")
        print(f"    최종 리워드: {best_config['final_reward']:.2f}")
        print(f"    파라미터:")
        for key, value in best_config['params'].items():
            if key != 'name':
                print(f"      {key}: {value}")

        print(f"\n💡 추천 다음 단계:")
        print(f"    1. 상위 3개 조합을 더 긴 학습(10000+ 업데이트)으로 재테스트")
        print(f"    2. 가장 좋은 조합 주변 파라미터 fine-tuning")
        print(f"    3. 네트워크 구조 개선 고려")

    else:
        print("❌ 성공한 테스트가 없습니다.")

    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()