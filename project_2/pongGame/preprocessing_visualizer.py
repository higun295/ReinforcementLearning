import numpy as np
import cv2
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import deque


class PreprocessingVisualizer:
    """PONG 게임 전처리 과정을 시각화하는 클래스"""

    def __init__(self):
        self.env = gym.make('ALE/Pong-v5', render_mode="rgb_array")

    def resize_frame(self, frame):
        """기존 전처리 함수 (preprocess_frame.py와 동일)"""
        frame = frame[30:-12, 5:-4]  # ROI 크롭핑
        frame = np.average(frame, axis=2)  # 그레이스케일 변환
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_NEAREST)
        frame = np.array(frame, dtype=np.uint8)
        return frame

    def get_sample_frames(self, num_frames=4):
        """샘플 게임 프레임들을 수집"""
        self.env.reset()
        frames = []

        # 몇 스텝 진행하여 의미있는 프레임 수집
        for _ in range(50):  # 게임이 시작될 때까지 대기
            frame, _, _, _, _ = self.env.step(0)  # NOOP

        # 실제 프레임 수집
        for i in range(num_frames):
            action = np.random.choice([0, 2, 3])  # 랜덤 액션
            for _ in range(3):  # 몇 스텝 간격
                frame, _, _, _, _ = self.env.step(action)
            frames.append(frame)

        return frames

    def create_preprocessing_comparison(self, save_path='preprocessing_comparison.png'):
        """전처리 전후 비교 시각화"""
        # 샘플 프레임 수집
        frames = self.get_sample_frames(4)
        original_frame = frames[0]

        # 전처리 단계별 결과
        steps = []

        # Step 1: 원본
        steps.append(('Original\n(210×160×3)', original_frame))

        # Step 2: 크롭핑
        cropped = original_frame[30:-12, 5:-4]
        steps.append(('Cropped\n(168×151×3)', cropped))

        # Step 3: 그레이스케일
        gray = np.average(cropped, axis=2)
        steps.append(('Grayscale\n(168×151×1)', gray))

        # Step 4: 리사이즈
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_NEAREST)
        steps.append(('Resized\n(84×84×1)', resized))

        # Step 5: 4프레임 스택 (마지막 프레임만 표시)
        processed_frames = [self.resize_frame(f) for f in frames]
        stacked = np.stack(processed_frames, axis=-1)
        steps.append(('4-Frame Stack\n(84×84×4)', stacked[:, :, -1]))  # 마지막 프레임만

        # 정규화된 버전
        normalized = stacked[:, :, -1] / 255.0
        steps.append(('Normalized\n(0~1 range)', normalized))

        # 플롯 생성
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('PONG 게임 데이터 전처리 과정', fontsize=16, fontweight='bold')

        axes = axes.flatten()

        for i, (title, image) in enumerate(steps):
            ax = axes[i]

            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB 이미지
                ax.imshow(image)
            else:
                # 그레이스케일 이미지
                ax.imshow(image, cmap='gray')

            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def create_4frame_stack_visualization(self, save_path='4frame_stack.png'):
        """4프레임 스택 과정 시각화"""
        frames = self.get_sample_frames(4)
        processed_frames = [self.resize_frame(f) for f in frames]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle('4프레임 스택을 통한 움직임 정보 포착', fontsize=16, fontweight='bold')

        for i, frame in enumerate(processed_frames):
            ax = axes[i]
            ax.imshow(frame, cmap='gray')
            ax.set_title(f'Frame t-{3 - i}', fontsize=12, fontweight='bold')
            ax.axis('off')

        # 하단에 설명 추가
        fig.text(0.5, 0.02,
                 '4개 연속 프레임을 스택하여 공과 패들의 움직임 방향 및 속도 정보를 제공',
                 ha='center', fontsize=12, style='italic')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        return fig

    def print_preprocessing_stats(self):
        """전처리 통계 정보 출력"""
        frames = self.get_sample_frames(1)
        original = frames[0]
        processed = self.resize_frame(original)

        print("=== PONG 게임 전처리 통계 ===")
        print(f"원본 이미지:")
        print(f"  - 크기: {original.shape}")
        print(f"  - 메모리: {original.nbytes:,} bytes ({original.nbytes / 1024:.1f} KB)")
        print(f"  - 데이터 타입: {original.dtype}")
        print(f"  - 픽셀 범위: {original.min()} ~ {original.max()}")

        print(f"\n전처리된 이미지:")
        print(f"  - 크기: {processed.shape}")
        print(f"  - 메모리: {processed.nbytes:,} bytes ({processed.nbytes / 1024:.1f} KB)")
        print(f"  - 데이터 타입: {processed.dtype}")
        print(f"  - 픽셀 범위: {processed.min()} ~ {processed.max()}")

        reduction = (1 - processed.nbytes / original.nbytes) * 100
        print(f"\n메모리 절약률: {reduction:.1f}%")

        # 4프레임 스택 메모리
        stack_memory = processed.nbytes * 4
        print(f"4프레임 스택 메모리: {stack_memory:,} bytes ({stack_memory / 1024:.1f} KB)")

    def close(self):
        """환경 정리"""
        self.env.close()


# 사용 예시
if __name__ == "__main__":
    # 시각화 객체 생성
    visualizer = PreprocessingVisualizer()

    # 전처리 과정 비교 이미지 생성
    print("전처리 과정 시각화 중...")
    visualizer.create_preprocessing_comparison('preprocessing_steps.png')

    # 4프레임 스택 시각화
    print("4프레임 스택 시각화 중...")
    visualizer.create_4frame_stack_visualization('4frame_stack.png')

    # 통계 정보 출력
    visualizer.print_preprocessing_stats()

    # 정리
    visualizer.close()

    print("\n✅ 시각화 완료!")
    print("생성된 파일:")
    print("  - preprocessing_steps.png: 전처리 단계별 비교")
    print("  - 4frame_stack.png: 4프레임 스택 과정")