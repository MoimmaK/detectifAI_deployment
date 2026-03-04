import { NextRequest, NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

export async function GET(
  request: NextRequest,
  { params }: { params: { videoId: string } }
) {
  try {
    const videoId = params.videoId
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000'

    // Don't follow redirects — the backend returns a 302 to a MinIO presigned URL.
    // We redirect the browser there directly instead of streaming through Vercel
    // (Vercel serverless has a ~4.5MB response body limit).
    const response = await fetch(`${API_URL}/api/v3/video/compressed/${videoId}`, {
      redirect: 'manual',
    })

    // Backend returns 302 with Location header → redirect browser to MinIO presigned URL
    if (response.status === 302 || response.status === 301) {
      const redirectUrl = response.headers.get('Location')
      if (redirectUrl) {
        return NextResponse.redirect(redirectUrl)
      }
    }

    // If the backend returned the video directly (non-redirect), try to stream it
    if (response.ok) {
      const videoStream = response.body
      if (!videoStream) {
        return NextResponse.json({ error: 'No video stream received' }, { status: 500 })
      }

      const responseHeaders = new Headers()
      responseHeaders.set('Content-Type', response.headers.get('Content-Type') || 'video/mp4')
      responseHeaders.set('Cache-Control', 'no-cache')
      const headersToForward = ['Content-Length', 'Content-Range', 'Accept-Ranges']
      headersToForward.forEach(header => {
        const value = response.headers.get(header)
        if (value) responseHeaders.set(header, value)
      })

      return new NextResponse(videoStream, { status: response.status, headers: responseHeaders })
    }

    // Error fallback
    let errorMessage = 'Failed to fetch compressed video'
    try {
      const ct = response.headers.get('content-type')
      if (ct && ct.includes('application/json')) {
        const errorData = await response.json()
        errorMessage = errorData.error || errorMessage
      }
    } catch { /* ignore parse errors */ }

    return NextResponse.json({ error: errorMessage }, { status: response.status })
  } catch (error) {
    console.error('❌ Error fetching compressed video:', error)
    return NextResponse.json(
      { error: `Failed to fetch compressed video: ${error instanceof Error ? error.message : 'Unknown error'}` },
      { status: 500 }
    )
  }
}