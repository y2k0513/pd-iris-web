const EYE_DEFINITIONS = Object.freeze({
  right: Object.freeze({
    outer: 33,
    inner: 133,
    upper: 159,
    lower: 145,
    irisCenter: 468,
    irisBoundary: [469, 470, 471, 472],
  }),

  left: Object.freeze({
    outer: 263,
    inner: 362,
    upper: 386,
    lower: 374,
    irisCenter: 473,
    irisBoundary: [474, 475, 476, 477],
  }),
});

const LIMBUS_TARGET_MIN_WIDTH = 300;
const LIMBUS_TARGET_MAX_WIDTH = 720;
const LIMBUS_UPSCALE = 5;
const LIMBUS_ANGLE_COUNT = 180;
const LIMBUS_MIN_POINTS = 26;
const LIMBUS_CONFIDENCE_THRESHOLD = 0.60;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function finitePoint(point) {
  return (
    point
    && Number.isFinite(point.x)
    && Number.isFinite(point.y)
  );
}

function pointToPixel(point, width, height) {
  return {
    x: point.x * width,
    y: point.y * height,
  };
}

function pointDistance(a, b) {
  return Math.hypot(
    a.x - b.x,
    a.y - b.y,
  );
}

function median(values) {
  const finite = values
    .filter(Number.isFinite)
    .sort((a, b) => a - b);

  if (!finite.length) return Number.NaN;

  const middle =
    Math.floor(finite.length / 2);

  return finite.length % 2
    ? finite[middle]
    : (
      finite[middle - 1]
      + finite[middle]
    ) / 2;
}

function percentile(values, fraction) {
  const finite = values
    .filter(Number.isFinite)
    .sort((a, b) => a - b);

  if (!finite.length) return Number.NaN;

  const index =
    clamp(fraction, 0, 1)
    * (finite.length - 1);

  const lower =
    Math.floor(index);

  const upper =
    Math.ceil(index);

  if (lower === upper) {
    return finite[lower];
  }

  const weight =
    index - lower;

  return finite[lower]
    * (1 - weight)
    + finite[upper]
    * weight;
}

function maxPairwiseDistance(points) {
  let maximum = 0;

  for (
    let first = 0;
    first < points.length;
    first += 1
  ) {
    for (
      let second = first + 1;
      second < points.length;
      second += 1
    ) {
      maximum = Math.max(
        maximum,
        pointDistance(
          points[first],
          points[second],
        ),
      );
    }
  }

  return maximum;
}

function calculateEyeRegion(
  landmarks,
  side,
  width,
  height,
) {
  const definition =
    EYE_DEFINITIONS[side];

  const indices = [
    definition.outer,
    definition.inner,
    definition.upper,
    definition.lower,
    definition.irisCenter,
    ...definition.irisBoundary,
  ];

  const points =
    indices.map(
      (index) => pointToPixel(
        landmarks[index],
        width,
        height,
      ),
    );

  const outer = pointToPixel(
    landmarks[definition.outer],
    width,
    height,
  );

  const inner = pointToPixel(
    landmarks[definition.inner],
    width,
    height,
  );

  const eyeWidth = Math.max(
    1,
    pointDistance(outer, inner),
  );

  const paddingX =
    eyeWidth * 0.48;

  const paddingY =
    eyeWidth * 0.42;

  const minX =
    Math.min(
      ...points.map((point) => point.x),
    ) - paddingX;

  const maxX =
    Math.max(
      ...points.map((point) => point.x),
    ) + paddingX;

  const minY =
    Math.min(
      ...points.map((point) => point.y),
    ) - paddingY;

  const maxY =
    Math.max(
      ...points.map((point) => point.y),
    ) + paddingY;

  const x = Math.floor(
    clamp(minX, 0, width - 2),
  );

  const y = Math.floor(
    clamp(minY, 0, height - 2),
  );

  const regionWidth = Math.max(
    2,
    Math.ceil(
      clamp(maxX, x + 2, width) - x,
    ),
  );

  const regionHeight = Math.max(
    2,
    Math.ceil(
      clamp(maxY, y + 2, height) - y,
    ),
  );

  return {
    x,
    y,
    width: regionWidth,
    height: regionHeight,
    eyeWidth,
  };
}

function createCrop(
  source,
  region,
) {
  const targetWidth =
    clamp(
      Math.round(
        region.width
        * LIMBUS_UPSCALE,
      ),
      LIMBUS_TARGET_MIN_WIDTH,
      LIMBUS_TARGET_MAX_WIDTH,
    );

  const targetHeight =
    Math.max(
      120,
      Math.round(
        region.height
        * targetWidth
        / region.width,
      ),
    );

  const canvas =
    document.createElement('canvas');

  canvas.width = targetWidth;
  canvas.height = targetHeight;

  const context =
    canvas.getContext(
      '2d',
      {
        willReadFrequently: true,
      },
    );

  context.drawImage(
    source,
    region.x,
    region.y,
    region.width,
    region.height,
    0,
    0,
    targetWidth,
    targetHeight,
  );

  return {
    canvas,
    scaleX:
      targetWidth / region.width,
    scaleY:
      targetHeight / region.height,
  };
}

function sampleNearest(
  data,
  width,
  height,
  x,
  y,
) {
  const px = clamp(
    Math.round(x),
    0,
    width - 1,
  );

  const py = clamp(
    Math.round(y),
    0,
    height - 1,
  );

  return data[
    py * width + px
  ];
}

function angleIsReliable(angle) {
  /*
   * 위·아래 눈꺼풀 방향은 limbus보다 눈꺼풀 경계가
   * 더 강하게 나타나는 경우가 많다. 수직축 주변 약 25°를
   * 제외하고 좌우 및 대각선 방향의 경계를 우선 사용한다.
   */
  return (
    Math.abs(Math.sin(angle)) < 0.90
  );
}

export function ellipseResidual(
  point,
  ellipse,
) {
  const angle =
    (
      Number(ellipse.angle) || 0
    ) * Math.PI / 180;

  const cos =
    Math.cos(angle);

  const sin =
    Math.sin(angle);

  const dx =
    point.x - ellipse.x;

  const dy =
    point.y - ellipse.y;

  const localX =
    dx * cos + dy * sin;

  const localY =
    -dx * sin + dy * cos;

  const radiusX =
    Math.max(
      1e-6,
      ellipse.width / 2,
    );

  const radiusY =
    Math.max(
      1e-6,
      ellipse.height / 2,
    );

  const normalizedRadius =
    Math.hypot(
      localX / radiusX,
      localY / radiusY,
    );

  return (
    Math.abs(normalizedRadius - 1)
    * (
      radiusX + radiusY
    ) / 2
  );
}

function fitEllipse(
  cv,
  points,
) {
  if (
    !Array.isArray(points)
    || points.length < 5
  ) {
    return null;
  }

  const values =
    points.flatMap(
      (point) => [
        Math.round(point.x),
        Math.round(point.y),
      ],
    );

  const matrix =
    cv.matFromArray(
      points.length,
      1,
      cv.CV_32SC2,
      values,
    );

  try {
    const fitted =
      cv.fitEllipse(matrix);

    if (
      !fitted?.center
      || !fitted?.size
    ) {
      return null;
    }

    const width =
      Number(fitted.size.width);

    const height =
      Number(fitted.size.height);

    if (
      !Number.isFinite(width)
      || !Number.isFinite(height)
      || width <= 0
      || height <= 0
    ) {
      return null;
    }

    return {
      x: Number(fitted.center.x),
      y: Number(fitted.center.y),
      width,
      height,
      angle:
        Number(fitted.angle) || 0,
    };
  } finally {
    matrix.delete();
  }
}

function robustFitEllipse(
  cv,
  points,
) {
  let inliers =
    [...points];

  let ellipse =
    fitEllipse(cv, inliers);

  for (
    let iteration = 0;
    iteration < 4;
    iteration += 1
  ) {
    if (
      !ellipse
      || inliers.length < LIMBUS_MIN_POINTS
    ) {
      break;
    }

    const residuals =
      inliers.map(
        (point) =>
          ellipseResidual(
            point,
            ellipse,
          ),
      );

    const center =
      median(residuals);

    const mad =
      median(
        residuals.map(
          (value) =>
            Math.abs(value - center),
        ),
      );

    const robustSigma =
      Number.isFinite(mad)
        ? mad * 1.4826
        : 0;

    const cutoff =
      Math.max(
        1.4,
        center
        + 2.2 * robustSigma,
      );

    const filtered =
      inliers.filter(
        (point) =>
          ellipseResidual(
            point,
            ellipse,
          ) <= cutoff,
      );

    if (
      filtered.length
      < LIMBUS_MIN_POINTS
      || filtered.length
        === inliers.length
    ) {
      break;
    }

    inliers = filtered;
    ellipse =
      fitEllipse(cv, inliers);
  }

  return {
    ellipse,
    inliers,
  };
}

function angularCoverage(
  points,
  center,
  binCount = 32,
) {
  if (!points.length) return 0;

  const bins =
    new Set();

  for (const point of points) {
    let angle =
      Math.atan2(
        point.y - center.y,
        point.x - center.x,
      );

    if (angle < 0) {
      angle += Math.PI * 2;
    }

    const bin =
      Math.min(
        binCount - 1,
        Math.floor(
          angle
          / (Math.PI * 2)
          * binCount,
        ),
      );

    bins.add(bin);
  }

  return bins.size / binCount;
}

function collectRadialCandidates({
  gray,
  gradientX,
  gradientY,
  center,
  minimumRadius,
  maximumRadius,
  upperY,
  lowerY,
}) {
  const candidates = [];

  const step =
    Math.max(
      0.8,
      (
        maximumRadius
        - minimumRadius
      ) / 90,
    );

  const contrastOffset =
    Math.max(
      1.5,
      (
        maximumRadius
        - minimumRadius
      ) * 0.035,
    );

  for (
    let index = 0;
    index < LIMBUS_ANGLE_COUNT;
    index += 1
  ) {
    const angle =
      index
      / LIMBUS_ANGLE_COUNT
      * Math.PI * 2;

    if (!angleIsReliable(angle)) {
      continue;
    }

    const cos =
      Math.cos(angle);

    const sin =
      Math.sin(angle);

    let best = null;

    for (
      let radius = minimumRadius;
      radius <= maximumRadius;
      radius += step
    ) {
      const x =
        center.x + cos * radius;

      const y =
        center.y + sin * radius;

      /*
       * 충분히 눈을 뜬 경우에만 눈꺼풀 안쪽의 점을 사용한다.
       * 눈이 작은 경우에는 각도 필터만 적용한다.
       */
      const aperture =
        lowerY - upperY;

      if (
        aperture > maximumRadius * 0.85
        && (
          y <= upperY + maximumRadius * 0.025
          || y >= lowerY - maximumRadius * 0.025
        )
      ) {
        continue;
      }

      if (
        x < 2
        || y < 2
        || x >= gray.cols - 2
        || y >= gray.rows - 2
      ) {
        continue;
      }

      const gx =
        sampleNearest(
          gradientX.data16S,
          gradientX.cols,
          gradientX.rows,
          x,
          y,
        );

      const gy =
        sampleNearest(
          gradientY.data16S,
          gradientY.cols,
          gradientY.rows,
          x,
          y,
        );

      const radialGradient =
        gx * cos + gy * sin;

      const inside =
        sampleNearest(
          gray.data,
          gray.cols,
          gray.rows,
          center.x
            + cos
            * (
              radius
              - contrastOffset
            ),
          center.y
            + sin
            * (
              radius
              - contrastOffset
            ),
        );

      const outside =
        sampleNearest(
          gray.data,
          gray.cols,
          gray.rows,
          center.x
            + cos
            * (
              radius
              + contrastOffset
            ),
          center.y
            + sin
            * (
              radius
              + contrastOffset
            ),
        );

      const contrast =
        outside - inside;

      /*
       * 홍채에서 공막 방향으로 나갈 때 밝아지는
       * 양의 radial gradient를 우선한다.
       */
      const score =
        Math.max(
          0,
          radialGradient,
        )
        + Math.max(
          -8,
          contrast,
        ) * 2.2;

      if (
        !best
        || score > best.score
      ) {
        best = {
          x,
          y,
          angle,
          radius,
          score,
          radialGradient,
          contrast,
        };
      }
    }

    if (best) {
      candidates.push(best);
    }
  }

  if (
    candidates.length
    < LIMBUS_MIN_POINTS
  ) {
    return candidates;
  }

  const threshold =
    Math.max(
      12,
      percentile(
        candidates.map(
          (point) => point.score,
        ),
        0.34,
      ),
    );

  let selected =
    candidates.filter(
      (point) =>
        point.score >= threshold
        && point.radialGradient > 0,
    );

  if (
    selected.length
    < LIMBUS_MIN_POINTS
  ) {
    selected =
      [...candidates]
        .sort(
          (a, b) =>
            b.score - a.score,
        )
        .slice(
          0,
          Math.max(
            LIMBUS_MIN_POINTS,
            Math.round(
              candidates.length * 0.55,
            ),
          ),
        );
  }

  return selected;
}

function toGlobalEllipse(
  ellipse,
  region,
  scaleX,
  scaleY,
) {
  const scale =
    (
      scaleX + scaleY
    ) / 2;

  return {
    x:
      region.x
      + ellipse.x / scaleX,

    y:
      region.y
      + ellipse.y / scaleY,

    width:
      ellipse.width / scale,

    height:
      ellipse.height / scale,

    angle:
      ellipse.angle,
  };
}

function toGlobalPoint(
  point,
  region,
  scaleX,
  scaleY,
) {
  return {
    x:
      region.x
      + point.x / scaleX,

    y:
      region.y
      + point.y / scaleY,
  };
}

function detectLimbusForEye({
  source,
  landmarks,
  side,
  width,
  height,
  cv,
  pupilHint = null,
  includeDebug = false,
}) {
  const definition =
    EYE_DEFINITIONS[side];

  const region =
    calculateEyeRegion(
      landmarks,
      side,
      width,
      height,
    );

  const crop =
    createCrop(
      source,
      region,
    );

  const scale =
    (
      crop.scaleX
      + crop.scaleY
    ) / 2;

  const mediaPipeCenterGlobal =
    pointToPixel(
      landmarks[
        definition.irisCenter
      ],
      width,
      height,
    );

  const initialCenterGlobal =
    finitePoint(
      pupilHint?.finalCenter,
    )
      ? pupilHint.finalCenter
      : (
        finitePoint(
          pupilHint?.ellipse,
        )
          ? {
            x: pupilHint.ellipse.x,
            y: pupilHint.ellipse.y,
          }
          : mediaPipeCenterGlobal
      );

  const center = {
    x:
      (
        initialCenterGlobal.x
        - region.x
      ) * crop.scaleX,

    y:
      (
        initialCenterGlobal.y
        - region.y
      ) * crop.scaleY,
  };

  const irisBoundaryGlobal =
    definition.irisBoundary.map(
      (index) => pointToPixel(
        landmarks[index],
        width,
        height,
      ),
    );

  const mediaPipeIrisDiameter =
    maxPairwiseDistance(
      irisBoundaryGlobal,
    );

  const irisRadius =
    Math.max(
      8,
      mediaPipeIrisDiameter
      * scale
      / 2,
    );

  const pupilRadius =
    pupilHint?.ellipse
      ? (
        (
          pupilHint.ellipse.width
          + pupilHint.ellipse.height
        ) / 4
      ) * scale
      : 0;

  const minimumRadius =
    Math.max(
      irisRadius * 0.64,
      pupilRadius > 0
        ? pupilRadius * 1.12
        : 0,
    );

  const maximumRadius =
    irisRadius * 1.34;

  const upperGlobal =
    pointToPixel(
      landmarks[definition.upper],
      width,
      height,
    );

  const lowerGlobal =
    pointToPixel(
      landmarks[definition.lower],
      width,
      height,
    );

  const upperY =
    (
      upperGlobal.y - region.y
    ) * crop.scaleY;

  const lowerY =
    (
      lowerGlobal.y - region.y
    ) * crop.scaleY;

  let sourceMat = null;
  let gray = null;
  let equalized = null;
  let blurred = null;
  let gradientX = null;
  let gradientY = null;

  try {
    sourceMat =
      cv.imread(crop.canvas);

    gray =
      new cv.Mat();

    cv.cvtColor(
      sourceMat,
      gray,
      cv.COLOR_RGBA2GRAY,
    );

    equalized =
      new cv.Mat();

    cv.equalizeHist(
      gray,
      equalized,
    );

    blurred =
      new cv.Mat();

    cv.GaussianBlur(
      equalized,
      blurred,
      new cv.Size(5, 5),
      0,
      0,
      cv.BORDER_DEFAULT,
    );

    gradientX =
      new cv.Mat();

    gradientY =
      new cv.Mat();

    cv.Sobel(
      blurred,
      gradientX,
      cv.CV_16S,
      1,
      0,
      3,
      1,
      0,
      cv.BORDER_DEFAULT,
    );

    cv.Sobel(
      blurred,
      gradientY,
      cv.CV_16S,
      0,
      1,
      3,
      1,
      0,
      cv.BORDER_DEFAULT,
    );

    const candidates =
      collectRadialCandidates({
        gray: blurred,
        gradientX,
        gradientY,
        center,
        minimumRadius,
        maximumRadius,
        upperY,
        lowerY,
      });

    const {
      ellipse,
      inliers,
    } =
      robustFitEllipse(
        cv,
        candidates,
      );

    if (!ellipse) {
      return {
        accepted: false,
        ellipse: null,
        confidence: 0,
        reason:
          '홍채 외곽 ellipse fitting 실패',
        candidateCount:
          candidates.length,
        inlierCount: 0,
      };
    }

    const majorAxis =
      Math.max(
        ellipse.width,
        ellipse.height,
      );

    const minorAxis =
      Math.min(
        ellipse.width,
        ellipse.height,
      );

    const axisRatio =
      minorAxis
      / Math.max(1, majorAxis);

    const diameterRatio =
      (
        majorAxis / scale
      )
      / Math.max(
        1,
        mediaPipeIrisDiameter,
      );

    const centerDistanceRatio =
      pointDistance(
        ellipse,
        {
          x:
            (
              mediaPipeCenterGlobal.x
              - region.x
            ) * crop.scaleX,

          y:
            (
              mediaPipeCenterGlobal.y
              - region.y
            ) * crop.scaleY,
        },
      ) / Math.max(
        1,
        irisRadius,
      );

    const coverage =
      angularCoverage(
        inliers,
        ellipse,
      );

    const inlierRatio =
      inliers.length
      / Math.max(
        1,
        candidates.length,
      );

    const edgeScore =
      clamp(
        (
          median(
            inliers.map(
              (point) => point.score,
            ),
          ) - 12
        ) / 240,
        0,
        1,
      );

    const centerScore =
      clamp(
        1
        - centerDistanceRatio
          / 0.45,
        0,
        1,
      );

    const diameterScore =
      clamp(
        1
        - Math.abs(
          diameterRatio - 1,
        ) / 0.32,
        0,
        1,
      );

    const axisScore =
      clamp(
        (
          axisRatio - 0.62
        ) / 0.34,
        0,
        1,
      );

    const confidence =
      clamp(
        coverage * 0.24
        + inlierRatio * 0.20
        + edgeScore * 0.22
        + centerScore * 0.16
        + diameterScore * 0.13
        + axisScore * 0.05,
        0,
        1,
      );

    const accepted = (
      inliers.length
        >= LIMBUS_MIN_POINTS
      && coverage >= 0.40
      && axisRatio >= 0.85
      && diameterRatio >= 0.84
      && diameterRatio <= 1.16
      && centerDistanceRatio <= 0.28
      && confidence
        >= LIMBUS_CONFIDENCE_THRESHOLD
    );

    const globalEllipse =
      toGlobalEllipse(
        ellipse,
        region,
        crop.scaleX,
        crop.scaleY,
      );

    return {
      accepted,
      ellipse:
        accepted
          ? globalEllipse
          : null,

      candidateEllipse:
        globalEllipse,

      confidence,
      reason:
        accepted
          ? 'radial-gradient-limbus'
          : '홍채 외곽 신뢰도 부족',

      candidateCount:
        candidates.length,

      inlierCount:
        inliers.length,

      diagnostics: {
        coverage,
        inlierRatio,
        edgeScore,
        centerDistanceRatio,
        diameterRatio,
        axisRatio,
        mediaPipeIrisDiameter,
      },

      debug:
        includeDebug
          ? {
            region,
            center:
              initialCenterGlobal,

            candidatePoints:
              candidates.map(
                (point) =>
                  toGlobalPoint(
                    point,
                    region,
                    crop.scaleX,
                    crop.scaleY,
                  ),
              ),

            inlierPoints:
              inliers.map(
                (point) =>
                  toGlobalPoint(
                    point,
                    region,
                    crop.scaleX,
                    crop.scaleY,
                  ),
              ),
          }
          : null,
    };
  } catch (error) {
    return {
      accepted: false,
      ellipse: null,
      confidence: 0,
      reason:
        `홍채 외곽 검출 오류: ${error.message}`,
      candidateCount: 0,
      inlierCount: 0,
    };
  } finally {
    for (
      const matrix of [
        sourceMat,
        gray,
        equalized,
        blurred,
        gradientX,
        gradientY,
      ]
    ) {
      if (matrix?.delete) {
        matrix.delete();
      }
    }
  }
}

export function refineIrisBoundaries({
  source,
  landmarks,
  width,
  height,
  cv = window.cv,
  pupilRefinement = null,
  includeDebug = false,
}) {
  if (
    !cv?.Mat
    || !Array.isArray(landmarks)
    || landmarks.length < 478
  ) {
    return {
      right: {
        accepted: false,
        ellipse: null,
        confidence: 0,
        reason:
          'OpenCV 또는 얼굴 랜드마크 없음',
      },

      left: {
        accepted: false,
        ellipse: null,
        confidence: 0,
        reason:
          'OpenCV 또는 얼굴 랜드마크 없음',
      },

      acceptedBoth: false,
    };
  }

  const right =
    detectLimbusForEye({
      source,
      landmarks,
      side: 'right',
      width,
      height,
      cv,
      pupilHint:
        pupilRefinement?.right,
      includeDebug,
    });

  const left =
    detectLimbusForEye({
      source,
      landmarks,
      side: 'left',
      width,
      height,
      cv,
      pupilHint:
        pupilRefinement?.left,
      includeDebug,
    });

  return {
    right,
    left,
    acceptedBoth:
      right.accepted
      && left.accepted,

    meanConfidence:
      (
        right.confidence
        + left.confidence
      ) / 2,
  };
}
