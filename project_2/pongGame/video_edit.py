import cv2
import numpy as np
import os
from datetime import datetime


class VideoSpeedController:
    """비디오 재생 배속을 조절하는 클래스"""

    def __init__(self, input_video_path):
        """
        Args:
            input_video_path: 입력 비디오 파일 경로
        """
        self.input_path = input_video_path

        # 입력 파일 존재 확인
        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {input_video_path}")

        # 비디오 캡처 객체 생성
        self.cap = cv2.VideoCapture(input_video_path)

        if not self.cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {input_video_path}")

        # 비디오 정보 읽기
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"📹 입력 비디오 정보:")
        print(f"   - 해상도: {self.width} x {self.height}")
        print(f"   - FPS: {self.fps:.2f}")
        print(f"   - 총 프레임: {self.total_frames}")
        print(f"   - 재생 시간: {self.total_frames / self.fps:.2f}초")

    def change_speed(self, speed_factor, output_filename=None):
        """
        비디오 재생 속도를 변경합니다.

        Args:
            speed_factor: 배속 (2.0 = 2배속, 0.5 = 0.5배속)
            output_filename: 출력 파일명 (None이면 자동 생성)

        Returns:
            str: 출력 파일 경로
        """
        if speed_factor <= 0:
            raise ValueError("배속은 0보다 커야 합니다.")

        # 출력 파일명 생성
        if output_filename is None:
            base_name = os.path.splitext(self.input_path)[0]
            timestamp = datetime.now().strftime('%H%M%S')
            output_filename = f"{base_name}_speed{speed_factor}x_{timestamp}.avi"

        print(f"\n🚀 배속 변경 시작:")
        print(f"   - 배속: {speed_factor}x")
        print(f"   - 출력 파일: {output_filename}")

        # 새로운 FPS 계산
        new_fps = self.fps * speed_factor

        # VideoWriter 설정
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_filename, fourcc, new_fps, (self.width, self.height))

        if not out.isOpened():
            raise RuntimeError("출력 비디오 파일을 생성할 수 없습니다.")

        # 프레임 처리
        frame_count = 0
        processed_frames = 0

        print("📊 처리 진행률:")

        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            # 프레임 쓰기
            out.write(frame)
            frame_count += 1
            processed_frames += 1

            # 진행률 표시 (10%마다)
            progress = (frame_count / self.total_frames) * 100
            if frame_count % max(1, self.total_frames // 10) == 0:
                print(f"   {progress:.0f}% 완료 ({frame_count}/{self.total_frames} 프레임)")

        # 정리
        out.release()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 처음으로 되돌리기

        # 결과 정보
        new_duration = processed_frames / new_fps
        original_duration = self.total_frames / self.fps

        print(f"\n✅ 배속 변경 완료!")
        print(f"   - 처리된 프레임: {processed_frames}")
        print(f"   - 새로운 FPS: {new_fps:.2f}")
        print(f"   - 원본 재생시간: {original_duration:.2f}초")
        print(f"   - 새로운 재생시간: {new_duration:.2f}초")

        # 파일 크기 확인
        if os.path.exists(output_filename):
            file_size = os.path.getsize(output_filename) / (1024 * 1024)  # MB
            print(f"   - 파일 크기: {file_size:.1f} MB")

        return output_filename

    def create_multiple_speeds(self, speed_factors=[2.0, 10.0]):
        """
        여러 배속의 비디오를 한 번에 생성합니다.

        Args:
            speed_factors: 배속 리스트

        Returns:
            list: 생성된 파일 경로들
        """
        output_files = []

        for speed in speed_factors:
            print(f"\n{'=' * 50}")
            output_file = self.change_speed(speed)
            output_files.append(output_file)

        print(f"\n🎬 모든 배속 비디오 생성 완료!")
        print(f"📁 생성된 파일들:")
        for i, file in enumerate(output_files):
            print(f"   {i + 1}. {file}")

        return output_files

    def preview_speeds(self, speed_factors=[1.0, 10.0]):
        """
        다양한 배속으로 미리보기 (실제 파일 생성 안함)

        Args:
            speed_factors: 미리볼 배속들
        """
        print(f"\n👀 배속 미리보기:")

        for speed in speed_factors:
            frame_skip = max(1, int(speed) - 1)  # 건너뛸 프레임 수
            delay = max(1, int(30 / speed))  # 프레임간 딜레이

            print(f"\n🎮 {speed}x 배속 미리보기 (ESC로 다음, Q로 종료)")

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 처음부터

            frame_count = 0
            while frame_count < 200:  # 200프레임만 미리보기
                ret, frame = self.cap.read()

                if not ret:
                    break

                # 배속에 따라 프레임 건너뛰기
                if frame_count % (frame_skip + 1) == 0:
                    cv2.imshow(f'Speed {speed}x Preview', frame)

                    key = cv2.waitKey(delay) & 0xFF
                    if key == 27:  # ESC
                        break
                    elif key == ord('q'):  # Q
                        cv2.destroyAllWindows()
                        return

                frame_count += 1

            cv2.destroyAllWindows()

    def close(self):
        """리소스 정리"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


# 사용 예시
if __name__ == "__main__":
    # 비디오 파일 경로
    video_path = "pong_gameplay_20250603_214732.avi"

    try:
        # VideoSpeedController 객체 생성
        speed_controller = VideoSpeedController(video_path)

        # 2배속과 3배속 비디오 생성
        output_files = speed_controller.create_multiple_speeds([2.0, 10.0])

        # 또는 개별 배속 생성
        # speed_2x = speed_controller.change_speed(2.0)
        # speed_3x = speed_controller.change_speed(3.0)

        # 미리보기 (선택사항)
        # speed_controller.preview_speeds([1.0, 2.0, 3.0])

        # 정리
        speed_controller.close()

    except FileNotFoundError as e:
        print(f"❌ 파일 오류: {e}")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")