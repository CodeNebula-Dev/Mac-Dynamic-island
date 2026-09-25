import SwiftUI

public struct MediaHUDView: View {
    public let title: String
    public let artist: String
    public let isPlaying: Bool

    @State private var barHeights: [CGFloat] = [0.3, 0.7, 0.4, 0.9, 0.5]
    private let timer = Timer.publish(every: 0.15, on: .main, in: .common).autoconnect()

    public init(title: String, artist: String, isPlaying: Bool) {
        self.title = title
        self.artist = artist
        self.isPlaying = isPlaying
    }

    public var body: some View {
        HStack(spacing: 12) {
            // Album Art placeholder / icon
            ZStack {
                RoundedRectangle(cornerRadius: 8)
                    .fill(
                        LinearGradient(
                            gradient: Gradient(colors: [.purple, .indigo, .pink]),
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 36, height: 36)

                Image(systemName: "music.note")
                    .foregroundColor(.white)
                    .font(.system(size: 16, weight: .bold))
            }

            // Song Info
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .foregroundColor(.white)
                    .lineLimit(1)

                Text(artist)
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(.gray)
                    .lineLimit(1)
            }

            Spacer()

            // Animated Audio Equalizer Bars
            HStack(alignment: .bottom, spacing: 2.5) {
                ForEach(0..<barHeights.count, id: \.self) { index in
                    RoundedRectangle(cornerRadius: 1.5)
                        .fill(Color.green)
                        .frame(width: 3, height: isPlaying ? max(4, barHeights[index] * 18) : 4)
                        .animation(.easeInOut(duration: 0.15), value: barHeights[index])
                }
            }
            .frame(width: 26, height: 18)
            .onReceive(timer) { _ in
                guard isPlaying else { return }
                barHeights = barHeights.map { _ in CGFloat.random(in: 0.2...1.0) }
            }
        }
        .padding(.horizontal, 16)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
