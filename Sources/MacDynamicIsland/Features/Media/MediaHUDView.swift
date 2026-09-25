import SwiftUI

public struct MediaHUDView: View {
    public let title: String
    public let artist: String
    public let isPlaying: Bool

    @State private var barHeights: [CGFloat] = [0.35, 0.75, 0.45, 0.90, 0.55]
    private let timer = Timer.publish(every: 0.15, on: .main, in: .common).autoconnect()

    public init(title: String, artist: String, isPlaying: Bool) {
        self.title = title
        self.artist = artist
        self.isPlaying = isPlaying
    }

    public var body: some View {
        HStack(spacing: 14) {
            // Album art / player icon
            ZStack {
                RoundedRectangle(cornerRadius: 10)
                    .fill(
                        LinearGradient(
                            gradient: Gradient(colors: [.purple, .indigo]),
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 38, height: 38)

                Image(systemName: isPlaying ? "music.note" : "pause.fill")
                    .foregroundColor(.white)
                    .font(.system(size: 16, weight: .bold))
            }

            VStack(alignment: .leading, spacing: 3) {
                Text(title)
                    .font(.system(size: 15, weight: .semibold, design: .rounded))
                    .foregroundColor(.white)
                    .lineLimit(1)

                Text(artist)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundColor(.gray)
                    .lineLimit(1)
            }

            Spacer()

            // Animated Equalizer Visualizer
            if isPlaying {
                HStack(alignment: .bottom, spacing: 3.5) {
                    ForEach(0..<barHeights.count, id: \.self) { index in
                        RoundedRectangle(cornerRadius: 1.5)
                            .fill(Color.green)
                            .frame(width: 4, height: max(5, barHeights[index] * 22))
                            .animation(.easeInOut(duration: 0.15), value: barHeights[index])
                    }
                }
                .frame(width: 32, height: 24)
                .onReceive(timer) { _ in
                    barHeights = barHeights.map { _ in CGFloat.random(in: 0.2...1.0) }
                }
            }
        }
        .padding(.horizontal, 8)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
